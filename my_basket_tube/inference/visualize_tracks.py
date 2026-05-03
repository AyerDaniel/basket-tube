"""
Renders tracked bounding boxes onto frames and exports a short MP4.

python my_basket_tube/inference/visualize_tracks.py
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# =============================================================================
# CONFIG
# =============================================================================
FRAMES_DIR  = Path("my_basket_tube/videos/game_1/frames")
TRACKS_CSV  = Path("my_basket_tube/csv/tracks.csv")
OUTPUT_PATH = Path("my_basket_tube/viz/tracks_sample.mp4")

START_FRAME = 1000   # first frame to render
NUM_FRAMES  = 500    # how many frames to render
FPS         = 25

# Colour per class
CLASS_COLORS = {
    'player': (0, 255, 0),    # green
    'ball':   (0, 128, 255),  # orange
}
DEFAULT_COLOR = (200, 200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS  = 2


# =============================================================================
# HELPERS
# =============================================================================

def color_for(class_, track_id):
    base = CLASS_COLORS.get(class_, DEFAULT_COLOR)
    # Tint by track_id hash so different tracks are visually distinct
    seed = hash(track_id) & 0xFFFFFF
    r = (base[0] + (seed & 0xFF)) % 256
    g = (base[1] + ((seed >> 8) & 0xFF)) % 256
    b = (base[2] + ((seed >> 16) & 0xFF)) % 256
    return (b, g, r)  # OpenCV is BGR


def draw_frame(img, frame_tracks):
    for _, row in frame_tracks.iterrows():
        x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
        color = color_for(row['class'], row['track_id'])

        cv2.rectangle(img, (x1, y1), (x2, y2), color, THICKNESS)

        player_number = row.get('player_number')
        team          = row.get('team')
        # Primary: player name lookup (stub — replace with roster dict)
        # Fallback: team + number, then just track_id
        if pd.notna(player_number) and player_number and pd.notna(team) and team:
            label = f"{team} #{player_number}"
        elif pd.notna(player_number) and player_number:
            label = f"#{player_number}"
        elif pd.notna(team) and team:
            label = str(team)
        else:
            label = str(row['track_id'])

        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), FONT, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)

    return img


# =============================================================================
# MAIN
# =============================================================================

def run():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading tracks...")
    df = pd.read_csv(TRACKS_CSV)
    all_frame_ids = sorted(df['frame_id'].unique().tolist())

    # Select window
    start_idx  = next((i for i, f in enumerate(all_frame_ids) if f >= START_FRAME), 0)
    window_ids = all_frame_ids[start_idx : start_idx + NUM_FRAMES]

    if not window_ids:
        print("No frames found in range.")
        return

    print(f"Rendering frames {window_ids[0]}–{window_ids[-1]} ({len(window_ids)} frames)...")

    # Determine output size from first frame
    sample_path = FRAMES_DIR / f"{window_ids[0]:04d}.jpg"
    sample_img  = cv2.imread(str(sample_path))
    if sample_img is None:
        raise FileNotFoundError(f"Cannot read {sample_path}")
    h, w = sample_img.shape[:2]

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS,
        (w, h)
    )

    df_window = df[df['frame_id'].isin(set(window_ids))]

    for frame_id in window_ids:
        img_path = FRAMES_DIR / f"{frame_id:04d}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Missing: {img_path.name}")
            continue

        frame_tracks = df_window[df_window['frame_id'] == frame_id]
        img = draw_frame(img, frame_tracks)

        # Frame counter overlay
        cv2.putText(img, f"frame {frame_id}", (10, 30), FONT, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    print(f"Done. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
