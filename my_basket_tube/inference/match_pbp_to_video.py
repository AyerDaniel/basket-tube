import os
import re
import cv2
import easyocr
import csv
from tqdm import tqdm

frames_dir  = 'my_basket_tube/videos/game_1/frames'
pbp_input   = 'my_basket_tube/csv/play_by_play.csv'
pbp_output  = 'my_basket_tube/csv/play_by_play_frames.csv'

clock_box   = (907, 968,  1054, 1017)
quarter_box = (947, 1011, 1076, 1056)
guard_box   = (891, 1014, 983,  1048)

quarter_anchors = {'1': 9481, '2': 49771, '3': 76531, '4': 114931}

ocr = easyocr.Reader(['en'], gpu=True)

def read_box(img, box):
    x1, y1, x2, y2 = box
    return ' '.join(ocr.readtext(img[y1:y2, x1:x2], detail=0)).strip()

def parse_clock(text):
    m = re.search(r'(\d{1,2})[:\.](\d{2})', text)
    return f"{int(m.group(1))}:{m.group(2)}" if m else None

def normalize_clock(raw):
    raw = raw.split('.')[0]
    if ':' not in raw:
        return f"0:{int(raw):02d}"
    m, s = map(int, raw.split(':'))
    return f"{m}:{s:02d}"

# Load events
with open(pbp_input) as f:
    reader = csv.DictReader(f)
    fieldnames = list(reader.fieldnames) + ['frame_id']
    rows = list(reader)

# Check how many rows already processed for resume
already_done = 0
if os.path.exists(pbp_output):
    with open(pbp_output) as f:
        already_done = sum(1 for _ in f) - 1  # subtract header
    open_mode = 'a'
    print(f'Resuming from row {already_done}')
else:
    open_mode = 'w'

# Sort frames numerically
frame_files = sorted(
    (f for f in os.listdir(frames_dir) if f.endswith('.jpg')),
    key=lambda f: int(re.search(r'\d+', f).group())
)
frame_ids = [int(re.search(r'\d+', f).group()) for f in frame_files]

current_idx = 0

with open(pbp_output, open_mode, newline='') as out_f:
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    if open_mode == 'w':
        writer.writeheader()

    for i, row in tqdm(enumerate(rows), total=len(rows), desc='Matching frames'):
        if i < already_done:
            continue

        quarter = str(row['quarter'])
        target  = normalize_clock(row['time'])
        anchor  = quarter_anchors[quarter]
        found   = None

        for j in range(current_idx, len(frame_ids), 30):
            if frame_ids[j] < anchor:
                continue

            img = cv2.imread(os.path.join(frames_dir, frame_files[j]))

            # Quarter check
            q_text  = read_box(img, quarter_box)
            g_text  = read_box(img, guard_box)
            q_match = re.search(r'(1st|2nd|3rd|4th)', q_text, re.IGNORECASE)
            if not q_match or g_text:
                continue
            q_num = {'1st':'1','2nd':'2','3rd':'3','4th':'4'}[q_match.group(1).lower()]
            if q_num != quarter:
                continue

            # Clock check
            clock = parse_clock(read_box(img, clock_box))
            if clock == target:
                found = frame_ids[j]
                current_idx = j
                break

        row['frame_id'] = found or ''
        writer.writerow(row)
        out_f.flush()

print(f'Done. Written to {pbp_output}')
