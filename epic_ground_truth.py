"""
epic_ground_truth.py
=====================
Carga las anotaciones oficiales de EPIC-KITCHENS-100 (EPIC_100_train.csv /
EPIC_100_validation.csv) y permite consultar la acción REAL (ground truth)
para un video_id y frame concretos, en vez de inferirla con CLIP.
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class GTSegment:
    verb: str
    verb_class: int
    noun: str
    noun_class: int
    start_frame: int
    stop_frame: int
    narration: str


class EpicGroundTruth:
    """
    Uso:
        gt = EpicGroundTruth(["EPIC_100_train.csv", "EPIC_100_validation.csv"])
        seg = gt.segment_at_frame("P01_11", 50)
        if seg:
            print(seg.verb, seg.noun)
    """

    REQUIRED_COLS = {
        "video_id", "verb", "verb_class", "noun", "noun_class",
        "start_frame", "stop_frame", "narration",
    }

    def __init__(self, csv_paths: List[str]):
        dfs = []
        for p in csv_paths:
            df = pd.read_csv(p)
            missing = self.REQUIRED_COLS - set(df.columns)
            if missing:
                raise ValueError(f"{p} no tiene las columnas esperadas: {missing}")
            dfs.append(df)

        self.df = pd.concat(dfs, ignore_index=True)

        # Índice por video_id, ordenado por start_frame, para lookup rápido
        self._by_video: Dict[str, pd.DataFrame] = {
            vid: g.sort_values("start_frame").reset_index(drop=True)
            for vid, g in self.df.groupby("video_id")
        }

    def has_video(self, video_id: str) -> bool:
        return video_id in self._by_video

    def segment_at_frame(self, video_id: str, frame_idx: int) -> Optional[GTSegment]:
        """Devuelve el segmento anotado que contiene frame_idx, o None si no hay ninguno
        (recordar: EPIC-KITCHENS solo anota ventanas de acción, no cada frame)."""
        g = self._by_video.get(video_id)
        if g is None:
            return None

        hit = g[(g.start_frame <= frame_idx) & (frame_idx <= g.stop_frame)]
        if hit.empty:
            return None

        row = hit.iloc[0]
        return GTSegment(
            verb=row.verb,
            verb_class=int(row.verb_class),
            noun=row.noun,
            noun_class=int(row.noun_class),
            start_frame=int(row.start_frame),
            stop_frame=int(row.stop_frame),
            narration=row.narration,
        )

    def segments_for_video(self, video_id: str) -> List[GTSegment]:
        """Todos los segmentos anotados de un video, en orden temporal."""
        g = self._by_video.get(video_id)
        if g is None:
            return []
        return [
            GTSegment(
                verb=r.verb, verb_class=int(r.verb_class),
                noun=r.noun, noun_class=int(r.noun_class),
                start_frame=int(r.start_frame), stop_frame=int(r.stop_frame),
                narration=r.narration,
            )
            for r in g.itertuples()
        ]

    def stats(self) -> Dict:
        return {
            "total_segments": len(self.df),
            "videos": self.df["video_id"].nunique(),
            "verb_classes": self.df["verb_class"].nunique(),
            "noun_classes": self.df["noun_class"].nunique(),
        }