"""
Two modes:
  1. SAMPLE mode (default): prints pixel counts for sampled detections to validate ranges
  2. PATCH mode: re-classifies all player rows in detections_all.csv using pixel counting

YOLOv8n-seg segments the player from the background. HSV color counting runs only
on the masked player pixels — no court floor, no crowd, no background contamination.

Usage:
  python correct_team_identification.py          # sample mode
  python correct_team_identification.py --patch  # patch detections_all.csv
"""

import sys
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

FRAMES_DIR     = Path("my_basket_tube/videos/game_1/frames")
DETECTIONS_CSV = Path("my_basket_tube/csv/detections_all.csv")
SAMPLES_PER_TEAM = 20

# HSV pixel-count ranges (OpenCV: H=0-179, S=0-255, V=0-255)
LAKERS_YELLOW = (( 15,  80,  80), ( 35, 255, 255))
LAKERS_PURPLE = ((125,  60,  40), (165, 255, 255))
WARRIORS_BLUE = (( 95,  60,  40), (130, 255, 255))

COCO_PERSON_CLASS = 0
BOX_PAD = 20  # padding around the detection box before passing to YOLO-seg

_seg_model = None


def get_seg_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = YOLO("yolov8n-seg.pt")
    return _seg_model


def get_player_mask(img: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    """Run YOLO-seg on a padded crop and return a boolean mask in crop coordinates.

    Returns (mask, cx1, cy1) where cx1/cy1 are the padded crop offsets into img,
    or (None, x1, y1) if no person was detected.
    """
    h, w = img.shape[:2]
    cx1 = max(0, x1 - BOX_PAD)
    cy1 = max(0, y1 - BOX_PAD)
    cx2 = min(w, x2 + BOX_PAD)
    cy2 = min(h, y2 + BOX_PAD)

    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None, cx1, cy1

    results = get_seg_model()(crop, conf=0.25, verbose=False)
    dets = sv.Detections.from_ultralytics(results[0])
    person_dets = dets[dets.class_id == COCO_PERSON_CLASS]

    if person_dets.mask is None or len(person_dets) == 0:
        return None, cx1, cy1

    # Pick the largest person mask — most likely the target player
    areas = [m.sum() for m in person_dets.mask]
    return person_dets.mask[int(np.argmax(areas))], cx1, cy1


def classify_crop(img: np.ndarray, row):
    x1 = max(0, int(row['x'] - row['width']  / 2))
    y1 = max(0, int(row['y'] - row['height'] / 2))
    x2 = min(img.shape[1], int(row['x'] + row['width']  / 2))
    y2 = min(img.shape[0], int(row['y'] + row['height'] / 2))

    if (y2 - y1) <= 0 or (x2 - x1) <= 0:
        return None, 0, 0

    mask, cx1, cy1 = get_player_mask(img, x1, y1, x2, y2)

    if mask is not None:
        # Mask is in padded-crop coordinates; apply it to the padded crop
        cx2 = cx1 + mask.shape[1]
        cy2 = cy1 + mask.shape[0]
        padded_crop = img[cy1:cy2, cx1:cx2].copy()
        padded_crop[~mask] = 0
        sample = padded_crop
    else:
        # Fallback: top half of the bounding box with 10% left/right margins
        bh = y2 - y1
        bw = x2 - x1
        sample = img[y1 : y1 + bh // 2,
                     x1 + int(bw * 0.10) : x2 - int(bw * 0.10)]

    if sample.size == 0:
        return None, 0, 0

    hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)

    yellow_mask = cv2.inRange(hsv, np.array(LAKERS_YELLOW[0]), np.array(LAKERS_YELLOW[1]))
    purple_mask = cv2.inRange(hsv, np.array(LAKERS_PURPLE[0]), np.array(LAKERS_PURPLE[1]))
    blue_mask   = cv2.inRange(hsv, np.array(WARRIORS_BLUE[0]), np.array(WARRIORS_BLUE[1]))

    lakers_count = int(yellow_mask.sum() / 255) + int(purple_mask.sum() / 255)
    blue_count   = int(blue_mask.sum()   / 255)

    total = lakers_count + blue_count
    team = 'warriors' if total > 0 and blue_count / total > .51 else 'lakers' #  Play with this threshold for better team detection.  Moving on for time.
    return team, lakers_count, blue_count


def sample_mode(df):
    players = df[df['class'] == 'player']

    for team in ['warriors', 'lakers']:
        print(f"\n--- {team.upper()} ---")
        subset = players[players['team'] == team]
        sample = subset.sample(min(SAMPLES_PER_TEAM, len(subset)), random_state=42)

        for _, row in sample.iterrows():
            img_path = FRAMES_DIR / f"{int(row['frame_id']):04d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            team_pred, lakers, blue = classify_crop(img, row)
            correct = '✓' if team_pred == team else '✗'
            print(f"  {correct} frame {int(row['frame_id']):6d}  (yello+purple))={lakers:6d}  blue={blue:6d}  → {team_pred}")


def patch_mode(df):
    players_mask = df['class'] == 'player'
    print(f"Re-classifying {players_mask.sum()} player detections...")

    teams = []
    frame_cache = {}

    for _, row in tqdm(df[players_mask].iterrows(), total=players_mask.sum()):
        fid = int(row['frame_id'])
        if fid not in frame_cache:
            img_path = FRAMES_DIR / f"{fid:04d}.jpg"
            frame_cache[fid] = cv2.imread(str(img_path))
            if len(frame_cache) > 50:
                oldest = next(iter(frame_cache))
                del frame_cache[oldest]

        img = frame_cache.get(fid)
        if img is None:
            teams.append(row.get('team'))
            continue

        team_pred, _, _ = classify_crop(img, row)
        teams.append(team_pred if team_pred else row.get('team'))

    df.loc[players_mask, 'team'] = teams
    df.to_csv(DETECTIONS_CSV, index=False)
    print(f"\nDone. Updated {DETECTIONS_CSV}")
    print(df[players_mask]['team'].value_counts())


if __name__ == "__main__":
    df = pd.read_csv(DETECTIONS_CSV)

    if '--patch' in sys.argv:
        patch_mode(df)
    else:
        sample_mode(df)
