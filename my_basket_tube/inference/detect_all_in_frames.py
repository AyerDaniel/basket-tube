"""
inference server start
python my_basket_tube/inference/detect_all_in_frames.py
"""

print("Make sure 'inference server start' is running before executing this script.")

import os
import json
import cv2
import numpy as np
import easyocr
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from inference_sdk import InferenceHTTPClient

# =============================================================================
# CONFIG
# =============================================================================
FRAMES_DIR      = Path("my_basket_tube/videos/game_1/frames")
OUTPUT_CSV      = Path("my_basket_tube/csv/detections_all.csv")
CHECKPOINT_PATH = Path("my_basket_tube/csv/detections_all_checkpoint.json")
MODEL_ID        = "basketball-hoop-ball-and-player/4"
API_KEY         = "YOUR_ROBOFLOW_API_KEY"
FLUSH_EVERY     = 500

# =============================================================================
# TEAM COLOR RANGES (HSV)
# Lakers: purple or gold | Warriors: blue or gold
# =============================================================================
TEAM_COLORS = {
    'lakers': [
        ((125, 50, 50),  (155, 255, 255)),   # purple
        ((20,  100, 100), (35, 255, 255)),    # gold
    ],
    'warriors': [
        ((100, 80, 80),  (125, 255, 255)),    # blue
        ((20,  100, 100), (35, 255, 255)),    # gold
    ]
}

# =============================================================================
# CLIENT + OCR
# =============================================================================

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key=API_KEY
)

reader = easyocr.Reader(['en'], gpu=True, verbose=False)

# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint():
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, 'r') as f:
            return set(json.load(f))
    return set()


def save_checkpoint(processed):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(list(processed), f)


# =============================================================================
# OCR — jersey number
# =============================================================================

def ocr_number(frame_bgr, x, y, w, h):
    x1 = max(0, int(x - w / 2))
    y1 = max(0, int(y - h / 2))
    x2 = min(frame_bgr.shape[1], int(x + w / 2))
    y2 = min(frame_bgr.shape[0], int(y + h / 2))

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Upscale small crops for better OCR accuracy
    if crop.shape[0] < 64:
        scale = 64 / crop.shape[0]
        crop  = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    results = reader.readtext(crop, allowlist='0123456789', detail=0)
    if results:
        # Keep only digit characters, return first result
        digits = ''.join(filter(str.isdigit, results[0]))
        return digits if digits else None
    return None


# =============================================================================
# TEAM CLASSIFICATION — dominant jersey color
# =============================================================================

def classify_team(frame_bgr, x, y, w, h):
    x1 = max(0, int(x - w / 2))
    y1 = max(0, int(y - h / 2))
    x2 = min(frame_bgr.shape[1], int(x + w / 2))
    y2 = min(frame_bgr.shape[0], int(y + h / 2))

    # Use upper-middle third of bounding box (jersey torso)
    torso_y1 = y1 + (y2 - y1) // 4
    torso_y2 = y1 + (y2 - y1) // 2
    crop = frame_bgr[torso_y1:torso_y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    scores = {}
    for team, ranges in TEAM_COLORS.items():
        mask_total = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            mask_total = cv2.bitwise_or(mask_total, mask)
        scores[team] = cv2.countNonZero(mask_total)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


# =============================================================================
# MAIN
# =============================================================================

def run():
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(FRAMES_DIR.glob("frame_*.jpg"))
    processed  = load_checkpoint()
    pending    = [f for f in all_frames if f.name not in processed]

    print(f"Total frames:      {len(all_frames)}")
    print(f"Already processed: {len(processed)}")
    print(f"Pending:           {len(pending)}")

    if not pending:
        print("All frames already processed.")
        return

    write_header = not OUTPUT_CSV.exists()
    csv_file     = open(OUTPUT_CSV, 'a', newline='')
    buffer       = []

    try:
        for frame_path in tqdm(pending, desc="Detecting"):
            frame_id  = int(frame_path.stem.split('_')[1])
            frame_bgr = cv2.imread(str(frame_path))

            try:
                result = client.infer(str(frame_path), model_id=MODEL_ID)

                for pred in result.get('predictions', []):
                    x, y, w, h       = pred['x'], pred['y'], pred['width'], pred['height']
                    cls              = pred['class']
                    pred['frame_id'] = frame_id
                    pred['detail']   = None
                    pred['team']     = None

                    if cls == 'number':
                        pred['detail'] = ocr_number(frame_bgr, x, y, w, h)

                    elif cls == 'player':
                        pred['team'] = classify_team(frame_bgr, x, y, w, h)

                    buffer.append(pred)

                processed.add(frame_path.name)

                if len(processed) % FLUSH_EVERY == 0:
                    df = pd.DataFrame(buffer)
                    df.to_csv(csv_file, index=False, header=write_header)
                    write_header = False
                    buffer = []
                    save_checkpoint(processed)

            except Exception as e:
                print(f"  Error on {frame_path.name}: {e}")
                continue

    finally:
        if buffer:
            df = pd.DataFrame(buffer)
            df.to_csv(csv_file, index=False, header=write_header)
        csv_file.close()
        save_checkpoint(processed)
        print(f"\nDone. Detections saved to {OUTPUT_CSV}")
        print(f"Processed: {len(processed)} / {len(all_frames)} frames")


if __name__ == "__main__":
    run()
