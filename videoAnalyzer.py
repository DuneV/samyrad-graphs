import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from tqdm import tqdm
from ultralytics import YOLO
import json
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from datetime import datetime


@dataclass
class Canonical:
    shape: str
    dimensions: Tuple[float, float, float]
    graspable: bool
    up_axis: str


class SpaceObject:
    def __init__(self, name):
        self.name = name
        self.pos = {}
        self.mask = None

    def update_pos(self, cx, cy, z0, label, canonical=None, scale=None):
        self.pos = {
            "name": label,
            "cx": cx,
            "cy": cy,
            "z0": float(z0),
            "z0_meters": float(z0) * scale if scale else None,
            "scale_m_per_px": scale,
            "dimensions": canonical.dimensions if canonical else None,
            "shape": canonical.shape if canonical else None,
            "up_axis": canonical.up_axis if canonical else None,
        }

    def update_mask(self, mask):
        self.mask = mask

    def save(self, name, run_name="prueba1"):
        Path(f"output/{run_name}").mkdir(parents=True, exist_ok=True)
        with open(
            f'output/{run_name}/{name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            "w",
        ) as f:
            json.dump(self.pos, f)


class CanonicalInstance:
    def __init__(self, path):
        self.path = path
        self.canonicals = {}

    def load_data(self):
        with open(self.path, "r") as f:
            data = json.load(f)
        for name, info in data["classes"].items():
            self.canonicals[name] = Canonical(
                shape=info["shape"],
                dimensions=(info["rx"], info["ry"], info["rz"]),
                graspable=info["graspable"],
                up_axis=info["up_axis"],
            )

    def get_classes(self) -> Dict[str, Canonical]:
        return self.canonicals


class VideoAnalyzer:

    def __init__(
        self,
        video,
        path,
        output_path,
        model_path,
        confidence,
        info_path,
        alpha,
        N=5,
        run_name="test_1",
        device="cuda",
        trainedAIP="depth-anything/Depth-Anything-V2-Small-hf",
        trainedAMDE="depth-anything/Depth-Anything-V2-Small-hf",
        store_depth_vectors: bool = False,
        max_stored_vectors: Optional[int] = 500,
    ):
        """
        store_depth_vectors : si True, guarda depth_np de cada frame en
            self.vectors para exportarlo luego con save(). En videos largos
            (decenas de miles de frames) esto puede consumir decenas de GB
            de RAM — por default está DESACTIVADO.
        max_stored_vectors  : si store_depth_vectors=True, límite de frames
            a mantener en memoria (ventana deslizante: se descartan los más
            viejos). None = sin límite (¡cuidado en videos largos!).
        """
        self.video = video
        self.path = path
        self.output = output_path
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(trainedAIP)
        self.modeldepth = AutoModelForDepthEstimation.from_pretrained(trainedAMDE).to(self.device)
        self.modelvision = YOLO(model_path).to(self.device)
        self.confidence = confidence
        self.vectors = []
        self.info_path = info_path
        self.alpha = alpha
        self.run_name = run_name
        self.N = N

        self.store_depth_vectors = store_depth_vectors
        self.max_stored_vectors = max_stored_vectors

        # Estado persistente entre frames
        self.confirmed: Dict[str, SpaceObject] = {}
        self._frame_counter: Dict[str, int] = {}
        self.scale: Optional[float] = None
        self.scale_samples: List[float] = []

    def load_info_canonical(self):
        instance = CanonicalInstance(self.info_path)
        instance.load_data()
        self.canonicals = instance.get_classes()

    def get_video_info(self):
        self.cap = cv2.VideoCapture(self.path)
        self.ret, self.frame = self.cap.read()
        self.rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output, fourcc, self.fps, (self.width, self.height))

    def process_frame(self, frame: np.ndarray, framecounter: int):
        """
        Procesa un único frame: YOLO + Depth-Anything.
        Actualiza self.confirmed con los objetos estables (>= N frames).
        Devuelve (yolo_result, depth_np).
        Llamado por pipeline() y por UnifiedPipeline.
        """
        self.frame = frame
        self.rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = self.processor(images=self.rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth = self.modeldepth(**inputs).predicted_depth
        self.depth_np = depth.squeeze().cpu().numpy()

        # Guardar el historial completo de profundidad por frame es MUY
        # costoso en RAM para videos largos. Desactivado por default
        # (store_depth_vectors=False). Si se activa, se usa una ventana
        # deslizante (max_stored_vectors) en vez de crecer sin límite.
        if self.store_depth_vectors:
            self.vectors.append(self.depth_np)
            if (self.max_stored_vectors is not None
                    and len(self.vectors) > self.max_stored_vectors):
                self.vectors.pop(0)

        result = self.modelvision(frame, conf=self.confidence, verbose=False)[0]
        seen_this_frame: set = set()

        if result.boxes is not None:
            for box in result.boxes:
                cx, cy, w_b, h_b = box.xywh[0].tolist()
                cls_id = int(box.cls.item())
                label = self.modelvision.names[cls_id]
                seen_this_frame.add(label)
                self._frame_counter[label] = self._frame_counter.get(label, 0) + 1

                if self._frame_counter[label] >= self.N:
                    just_confirmed = label not in self.confirmed
                    if just_confirmed:
                        self.confirmed[label] = SpaceObject(f"{label}-frame{framecounter}")

                    dh, dw = self.depth_np.shape
                    fh, fw = frame.shape[:2]
                    cx_d = max(0, min(int(cx * dw / fw), dw - 1))
                    cy_d = max(0, min(int(cy * dh / fh), dh - 1))
                    z0 = self.depth_np[cy_d, cx_d]

                    radius = max(w_b, h_b) / 2
                    centroide_d = (cx_d, cy_d)
                    yolo_mesh = self.create_mask("sphere", centroide_d, self.depth_np, [radius, 0, 0])

                    if label in self.canonicals:
                        canonical = self.canonicals[label]
                        real_size = max(canonical.dimensions[0], canonical.dimensions[1])
                        pixel_size = max(w_b, h_b)
                        self.scale_samples.append(real_size / pixel_size)
                        self.scale = np.mean(self.scale_samples)
                        filter_mesh = self.convexmesh(self.alpha, yolo_mesh, canonical, centroide_d, label)
                        self.confirmed[label].update_pos(cx, cy, z0, label, canonical, self.scale)
                    else:
                        filter_mesh = yolo_mesh
                        self.confirmed[label].update_pos(cx, cy, z0, label, scale=self.scale)

                    self.confirmed[label].update_mask(filter_mesh)

                    # Antes esto se guardaba en CADA frame una vez confirmado
                    # el objeto (con timestamp único), generando cientos de
                    # miles de archivos en videos largos. Ahora solo se
                    # guarda al confirmarse por primera vez; las posiciones
                    # actualizadas quedan reflejadas en los checkpoints
                    # periódicos de UnifiedPipeline (confirmed_objects.json).
                    if just_confirmed:
                        self.confirmed[label].save(label, self.run_name)

        for lbl in list(self._frame_counter):
            if lbl not in seen_this_frame:
                self._frame_counter[lbl] = 0

        return result, self.depth_np

    def pipeline(self):
        self.get_video_info()
        self.load_info_canonical()

        framecounter = 0
        with tqdm(total=self.total_frames, desc="VideoAnalyzer", unit="frame") as pbar:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                yolo_result, depth_np = self.process_frame(frame, framecounter)

                annotated = yolo_result.plot()
                self.out.write(annotated)
                pbar.update(1)
                framecounter += 1

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

    def save(self, name):
        if not self.vectors:
            print("(store_depth_vectors=False o sin frames acumulados: "
                  "no se guarda depth_vectors.npy)")
            return
        arr = np.array(self.vectors)
        np.save(name, arr)

    def create_mask(self, croptype, *kargs):
        match croptype:
            case "sphere":
                return self.get_sphere_mask(*kargs)
            case "box":
                return self.get_box_mask(*kargs)
            case "cylinder":
                return self.get_cylinder_mask(*kargs)
            case _:
                print(f"Warning: Shape '{croptype}' not available")

    @staticmethod
    def get_sphere_mask(center, depth_cloud, dim):
        radius, ry, rz = dim
        h, w = depth_cloud.shape
        cx, cy = center
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        zz = depth_cloud
        z0 = depth_cloud[int(cy), int(cx)]
        dist = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - z0) ** 2
        return (dist <= radius ** 2).astype(np.uint8)

    @staticmethod
    def get_box_mask(center, depth_cloud, dim):
        LA, LB, LC = dim
        h, w = depth_cloud.shape
        cx, cy = center
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        zz = depth_cloud
        z0 = depth_cloud[int(cy), int(cx)]
        return (
            (np.abs(xx - cx) <= LA / 2)
            & (np.abs(yy - cy) <= LB / 2)
            & (np.abs(zz - z0) <= LC / 2)
        ).astype(np.uint8)

    @staticmethod
    def get_cylinder_mask(center, depth_cloud, dim, fig):
        radius, _, LA = dim
        h, w = depth_cloud.shape
        cx, cy = center
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        zz = depth_cloud
        z0 = depth_cloud[int(cy), int(cx)]
        if fig.up_axis == "z":
            dist = (xx - cx) ** 2 + (yy - cy) ** 2
            return ((np.abs(zz - z0) <= LA / 2) & (dist <= radius ** 2)).astype(np.uint8)
        elif fig.up_axis == "x":
            dist = (yy - cy) ** 2 + (zz - z0) ** 2
            return ((np.abs(xx - cx) <= LA / 2) & (dist <= radius ** 2)).astype(np.uint8)
        elif fig.up_axis == "y":
            # Eje "arriba" es y (imagen 2D): el cilindro se extiende en yy,
            # la sección circular queda en el plano xx-zz (profundidad).
            dist = (xx - cx) ** 2 + (zz - z0) ** 2
            return ((np.abs(yy - cy) <= LA / 2) & (dist <= radius ** 2)).astype(np.uint8)
        else:
            # Solo avisa UNA vez por valor desconocido, no en cada frame
            key = f"_warned_axis_{fig.up_axis}"
            if not getattr(VideoAnalyzer, key, False):
                print(f"Warning not found: {fig.up_axis}")
                setattr(VideoAnalyzer, key, True)
            return None

    def convexmesh(self, alpha, mesh, object, center, name):
        if object.dimensions[0] is not None:
            if object.shape == "cylinder":
                initial_guest = self.create_mask(object.shape, center, self.depth_np, object.dimensions, object)
            else:
                initial_guest = self.create_mask(object.shape, center, self.depth_np, object.dimensions)
            if initial_guest is None:
                return mesh
            return initial_guest * (1 - alpha) + alpha * mesh
        return mesh

    @classmethod
    def flag_class(cls, finalizado):
        cls.finalizado = finalizado


if __name__ == "__main__":
    base_folder = Path(__file__).resolve().parent
    ws = base_folder.parent.parent
    analyzer = VideoAnalyzer(
        video="test.mp4",
        path=str(base_folder / "test.mp4"),
        output_path=str(base_folder / "output.mp4"),
        model_path=str(ws / "yoloe-11l-seg-pf.pt"),
        confidence=0.5,
        info_path=str(ws / "src/ar_perception/models3d/canonical.json"),
        alpha=0.5,
        run_name="prueba1",
    )
    analyzer.pipeline()
    analyzer.save("depth_vectors")