#!/usr/bin/env python3
"""
process_chunk.py
=================
Procesa un RANGO de frames de un video (no el video completo) y guarda
el resultado en disco. Pensado para correr como subproceso aislado:
al terminar, el proceso muere y el sistema operativo libera TODA su
memoria — incluidas fugas nativas (MediaPipe, YOLO, CLIP) que no se
liberan correctamente dentro de un mismo proceso Python de larga
duración.

No se llama directamente en uso normal: lo invoca chunked_pipeline.py
una vez por cada chunk.
"""

import argparse
import json
from pathlib import Path

import cv2
import torch

from videoAnalyzer import VideoAnalyzer
from storyTelling import StoryTelling
from unifiedPipeline2 import UnifiedPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--end-frame", type=int, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--info-path", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--N", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_run_name = f"{args.run_name}_chunk{args.chunk_id}"

    va = VideoAnalyzer(
        video=args.video, path=args.video,
        output_path=str(out_dir / f"chunk_{args.chunk_id}_annotated.mp4"),
        model_path=args.model_path, confidence=args.confidence,
        info_path=args.info_path, alpha=0.5, N=args.N,
        run_name=chunk_run_name, device=device,
        store_depth_vectors=False,
    )
    st = StoryTelling(
        video=args.video, path=args.video,
        output_path=str(out_dir / f"chunk_{args.chunk_id}_annotated.mp4"),
        model_path=args.model_path, confidence=args.confidence,
        N=args.N, run_name=chunk_run_name,
        vlm_backend="clip", device=device,
    )

    va.get_video_info()
    va.load_info_canonical()
    st.fps, st.width, st.height = va.fps, va.width, va.height

    # Salta directo al frame de inicio del chunk (sin re-procesar lo anterior)
    va.cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    framecounter = args.start_frame
    while va.cap.isOpened() and framecounter < args.end_frame:
        ret, frame = va.cap.read()
        if not ret:
            break

        yolo_result, depth_np = va.process_frame(frame, framecounter)
        frame_data = st.process_frame_hands(frame, yolo_result, framecounter)
        if frame_data is not None:
            st.scene.append(frame_data)

        annotated = yolo_result.plot(img=frame.copy())
        va.out.write(annotated)

        framecounter += 1

    va.cap.release()
    va.out.release()

    # Reutiliza la lógica de pesos ya probada de UnifiedPipeline en vez
    # de reimplementarla — generate_scenario() solo lee self.va.confirmed
    # y self.st.semantic_line, no necesita que se haya llamado a run().
    pipeline = UnifiedPipeline(va, st, checkpoint_every=None, memory_log_every=None)
    scenario = pipeline.generate_scenario(goal=None)
    scenario["semantic_line"] = st.semantic_line
    scenario["edge_labels"] = {str(k): v for k, v in scenario["edge_labels"].items()}

    result = {
        "chunk_id":     args.chunk_id,
        "start_frame":  args.start_frame,
        "end_frame":    framecounter,
        "scenario":     scenario,
    }
    with open(out_dir / f"chunk_{args.chunk_id}.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"chunk {args.chunk_id} ({args.start_frame}-{framecounter}): "
          f"{len(va.confirmed)} objetos, {len(st.semantic_line)} eventos")


if __name__ == "__main__":
    main()