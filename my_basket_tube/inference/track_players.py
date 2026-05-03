"""
Run after detect_all_in_frames.py completes.
Lightweight IoU + Hungarian assignment tracker — no GPU required.

python my_basket_tube/inference/track_players.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# =============================================================================
# CONFIG
# =============================================================================
DETECTIONS_CSV  = Path("my_basket_tube/csv/detections_all.csv")
OUTPUT_CSV      = Path("my_basket_tube/csv/tracks.csv")
CHECKPOINT_PATH = Path("my_basket_tube/csv/tracks_checkpoint.json")

IOU_THRESHOLD = 0.3   # minimum IoU to match a detection to an existing track
MAX_MISSES    = 3     # frames a track survives without a matched detection
FLUSH_EVERY   = 5000  # rows between CSV flushes

TRACK_CLASSES = {'player', 'ball'}


# =============================================================================
# HELPERS
# =============================================================================

def bbox_from_xywh(x, y, w, h):
    return (int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2))


def iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return inter / (a1 + a2 - inter)


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r') as f:
            data = json.load(f)
        return data.get('last_frame_id', 0), data.get('segment_id', 0)
    return 0, 0


def save_checkpoint(last_frame_id, segment_id):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({'last_frame_id': int(last_frame_id), 'segment_id': int(segment_id)}, f)


# =============================================================================
# TRACKER
# =============================================================================

class Track:
    def __init__(self, track_id, segment_id, class_, team, detail, bbox):
        self.track_id   = track_id
        self.segment_id = segment_id
        self.class_     = class_
        self.team       = team
        self.detail     = detail
        self.bbox       = bbox
        self.misses     = 0


class IoUTracker:
    def __init__(self, iou_thresh=IOU_THRESHOLD, max_misses=MAX_MISSES):
        self.tracks     = []
        self.next_id    = 0
        self.segment_id = 0
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses

    def reset(self):
        self.tracks     = []
        self.next_id    = 0
        self.segment_id += 1

    def _match(self, detections):
        if not self.tracks or not detections:
            return [], list(range(len(self.tracks))), list(range(len(detections)))

        cost = np.zeros((len(self.tracks), len(detections)))
        for t, track in enumerate(self.tracks):
            for d, det in enumerate(detections):
                cost[t, d] = iou(track.bbox, det['bbox'])

        row_ind, col_ind = linear_sum_assignment(-cost)

        matched, unmatched_t, unmatched_d = [], list(range(len(self.tracks))), list(range(len(detections)))
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= self.iou_thresh:
                matched.append((r, c))
                unmatched_t.remove(r)
                unmatched_d.remove(c)

        return matched, unmatched_t, unmatched_d

    def update(self, frame_id, detections):
        if not detections:
            self.reset()
            return []

        matched, unmatched_t, unmatched_d = self._match(detections)
        rows = []

        for t_idx, d_idx in matched:
            track = self.tracks[t_idx]
            det   = detections[d_idx]
            track.bbox   = det['bbox']
            track.misses = 0
            rows.append(self._row(frame_id, track, det['confidence']))

        for t_idx in unmatched_t:
            self.tracks[t_idx].misses += 1

        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        for d_idx in unmatched_d:
            det   = detections[d_idx]
            track = Track(
                track_id   = f"{self.segment_id}_{self.next_id}",
                segment_id = self.segment_id,
                class_     = det['class'],
                team       = det['team'],
                detail     = det['detail'],
                bbox       = det['bbox']
            )
            self.next_id += 1
            self.tracks.append(track)
            rows.append(self._row(frame_id, track, det['confidence']))

        return rows

    def _row(self, frame_id, track, confidence):
        return {
            'frame_id':      frame_id,
            'segment_id':    track.segment_id,
            'track_id':      track.track_id,
            'class':         track.class_,
            'x1':            track.bbox[0],
            'y1':            track.bbox[1],
            'x2':            track.bbox[2],
            'y2':            track.bbox[3],
            'confidence':    confidence,
            'team':          track.team,
            'player_number': track.detail
        }


# =============================================================================
# MAIN
# =============================================================================

def run():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    print("Loading detections...")
    df = pd.read_csv(DETECTIONS_CSV)
    df = df[df['class'].isin(TRACK_CLASSES)]
    all_frame_ids = sorted(df['frame_id'].unique().tolist())

    last_frame_id, segment_id = load_checkpoint()
    start_idx = next(
        (i for i, f in enumerate(all_frame_ids) if f > last_frame_id),
        len(all_frame_ids)
    )

    print(f"Total frames with detections: {len(all_frame_ids)}")
    print(f"Resuming from frame {last_frame_id}, segment {segment_id}")
    print(f"Pending: {len(all_frame_ids) - start_idx} frames")

    if start_idx >= len(all_frame_ids):
        print("Tracking complete.")
        return

    tracker             = IoUTracker()
    tracker.segment_id  = segment_id

    write_header = not OUTPUT_CSV.exists()
    csv_file     = open(OUTPUT_CSV, 'a', newline='')
    buffer       = []

    try:
        for frame_id in tqdm(all_frame_ids[start_idx:], desc="Tracking"):
            frame_df   = df[df['frame_id'] == frame_id]
            detections = [
                {
                    'bbox':       bbox_from_xywh(r['x'], r['y'], r['width'], r['height']),
                    'class':      r['class'],
                    'team':       r.get('team'),
                    'detail':     r.get('detail'),
                    'confidence': r['confidence']
                }
                for _, r in frame_df.iterrows()
            ]

            buffer.extend(tracker.update(frame_id, detections))

            if len(buffer) >= FLUSH_EVERY:
                pd.DataFrame(buffer).to_csv(csv_file, index=False, header=write_header)
                write_header = False
                buffer = []
                save_checkpoint(frame_id, tracker.segment_id)

    finally:
        if buffer:
            pd.DataFrame(buffer).to_csv(csv_file, index=False, header=write_header)
        csv_file.close()
        save_checkpoint(
            all_frame_ids[-1] if all_frame_ids else last_frame_id,
            tracker.segment_id
        )
        print(f"\nTracking complete. Output: {OUTPUT_CSV}")


if __name__ == "__main__":
    run()
