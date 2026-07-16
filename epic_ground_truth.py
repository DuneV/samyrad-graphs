# epic_ground_truth.py
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

class EpicGroundTruth:

    def __init__(self, csv_paths: List[str]):
        dfs = [pd.read_csv(p) for p in csv_paths]
        self.df = pd.concat(dfs, ignore_index=True)
        self._by_video: Dict[str, pd.DataFrame] = {
            vid: g.sort_values("start_frame")
            for vid, g in self.df.groupby("video_id")
        }

    def segment_at_frame(self, video_id: str, frame_idx: int) -> Optional[GTSegment]:
        g = self._by_video.get(video_id)
        if g is None:
            return None
        hit = g[(g.start_frame <= frame_idx) & (frame_idx <= g.stop_frame)]
        if hit.empty:
            return None
        row = hit.iloc[0]
        return GTSegment(
            verb=row.verb, verb_class=int(row.verb_class),
            noun=row.noun, noun_class=int(row.noun_class),
            start_frame=int(row.start_frame), stop_frame=int(row.stop_frame),
        )
    
if __name__ == '__main__':
    gt = EpicGroundTruth(["annotations/EPIC_100_train.csv"])
    print(gt.df.head())
    seg = gt.segment_at_frame("P01_01", 1000)
    print(seg)