"""
chunked_pipeline.py
====================
Procesa un video largo en chunks de tamaño acotado (por número de
frames), cada uno en un SUBPROCESO separado. Cuando un chunk termina,
su proceso muere y el sistema operativo recupera TODA la RAM/VRAM que
usó — incluidas fugas nativas que no se liberan dentro del mismo
proceso Python (ver el workaround de HandLandmarker en storyTelling.py,
que ya no alcanza por sí solo para videos muy largos).

Combina los resultados de todos los chunks en un único scenario dict,
compatible con collect_training_data.py / validate_scenario() /
train_supervised().

Uso:
    from chunked_pipeline import process_video_chunked

    scenario = process_video_chunked(
        video_path  = "kitchen/EPIC-KITCHENS/P01/videos/P01_101.MP4",
        run_name    = "P01_101",
        model_path  = "yoloe-11l-seg-pf.pt",
        info_path   = "canonical.json",
        chunk_frames = 5000,
    )
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2


def get_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def process_video_chunked(
    video_path: str,
    run_name: str,
    model_path: str,
    info_path: str,
    chunk_frames: int = 5000,
    confidence: float = 0.5,
    N: int = 5,
    chunks_dir: Optional[str] = None,
    resume: bool = True,
) -> Dict:
    """
    Divide el video en tramos de `chunk_frames` frames y procesa cada
    uno en un subproceso aislado (process_chunk.py).

    chunk_frames : tamaño de cada tramo. Referencia aproximada (720p,
        YOLO+Depth+CLIP+MediaPipe activos): 5000 frames ronda entre
        300MB y 1.5GB de pico de RAM por chunk según cuántas manos/
        objetos haya en pantalla — ajusta según la RAM real disponible.
    resume : si True (default), un chunk cuyo JSON ya existe se salta.
        Deja retomar un video interrumpido sin reprocesar todo.
    """
    total_frames = get_total_frames(video_path)
    chunks_dir_path = Path(chunks_dir or f"chunks/{run_name}")
    chunks_dir_path.mkdir(parents=True, exist_ok=True)

    boundaries = list(range(0, total_frames, chunk_frames))
    print(f"{run_name}: {total_frames} frames -> {len(boundaries)} chunks "
          f"de hasta {chunk_frames} frames cada uno")

    script_path = Path(__file__).resolve().parent / "process_chunk.py"

    for chunk_id, start in enumerate(boundaries):
        end = min(start + chunk_frames, total_frames)
        chunk_json = chunks_dir_path / f"chunk_{chunk_id}.json"

        if resume and chunk_json.exists():
            print(f"  chunk {chunk_id} ({start}-{end}): ya existe, se salta")
            continue

        print(f"  chunk {chunk_id} ({start}-{end}): procesando en subproceso...")
        cmd = [
            sys.executable, str(script_path),
            "--video", video_path,
            "--start-frame", str(start),
            "--end-frame", str(end),
            "--model-path", model_path,
            "--info-path", info_path,
            "--run-name", run_name,
            "--chunk-id", str(chunk_id),
            "--output-dir", str(chunks_dir_path),
            "--confidence", str(confidence),
            "--N", str(N),
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  ⚠ chunk {chunk_id} terminó con error "
                  f"(code {result.returncode}), se continúa con el resto")

    return merge_chunks(chunks_dir_path)


def merge_chunks(chunks_dir: Path) -> Dict:
    """
    Combina todos los chunk_N.json de una carpeta en un único scenario.
    - detected_objects / object_positions: unión; si un objeto aparece
      en varios chunks, se queda con la posición del chunk MÁS AVANZADO
      (más representativa del estado final del objeto en el video).
    - edge_labels: unión de todos los chunks (cada chunk ya calculó sus
      pesos correctamente vía UnifiedPipeline.generate_scenario).
    - required_actions / target_objects: unión.
    - semantic_line se concatena completa, en orden de chunk.
    """
    chunk_files = sorted(chunks_dir.glob("chunk_*.json"),
                          key=lambda p: int(p.stem.split("_")[1]))
    if not chunk_files:
        raise FileNotFoundError(f"No hay chunk_*.json en {chunks_dir}")

    detected_objects: Dict[str, tuple] = {}
    edge_labels: Dict[tuple, float] = {}
    required_actions: set = set()
    target_objects: set = set()
    all_semantic_line: List[dict] = []
    first_goal = None

    for cf in chunk_files:
        data = json.loads(cf.read_text())
        sc = data["scenario"]

        for obj, pos in sc.get("object_positions", {}).items():
            detected_objects[obj] = pos   # el último chunk que lo vea "gana"

        # Cada chunk guarda edge_labels con claves string (JSON no soporta
        # tuples como clave). Las reconstruimos a tuple real acá, para que
        # el resultado final sea consistente con scenario_from_video()
        # (no-chunked) y con lo que espera validate_scenario().
        for k_str, v in sc.get("edge_labels", {}).items():
            inner = k_str.strip("()").replace("'", "").split(", ")
            if len(inner) == 3:
                edge_labels[tuple(inner)] = v
            else:
                print(f"  ⚠ clave de edge_labels con formato inesperado, se ignora: {k_str!r}")

        required_actions.update(sc.get("required_actions", []))
        target_objects.update(sc.get("target_objects", []))
        all_semantic_line.extend(sc.get("semantic_line", []))

        if first_goal is None and sc.get("goal") not in (None, "Perform robot task"):
            first_goal = sc["goal"]

    return {
        "goal":             first_goal or "Perform robot task",
        "target_objects":   list(target_objects),
        "required_actions": list(required_actions),
        "detected_objects": list(detected_objects.keys()),
        "object_positions": detected_objects,
        "edge_labels":      edge_labels,   # ya viene con claves string "(a, b, c)"
        "semantic_line":    all_semantic_line,
        "n_chunks":         len(chunk_files),
    }