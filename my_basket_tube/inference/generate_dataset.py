import os
import json
import time
import base64
import anthropic
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RETRY_DELAYS = [10, 20, 40, 60, 300]  # seconds between attempts

# =============================================================================
# CONFIG
# =============================================================================
CONTEXT_SECONDS = 5
FPS             = 30
SUBTITLES_CSV   = "my_basket_tube/csv/subtitles.csv"
FRAMES_DIR      = Path("my_basket_tube/videos/game_1/frames")
DATASET_PATH    = Path("my_basket_tube/csv/training_dataset.json")
CLAUDE_MODEL    = "claude-sonnet-4-6"

# =============================================================================
# HELPERS
# =============================================================================

def frame_path(frame_idx):
    return FRAMES_DIR / f"frame_{frame_idx}.jpg"


def encode_image(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def format_timestamp(frame_idx):
    seconds = frame_idx / FPS
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def build_commentary_transcript(df_subtitles):
    lines = []
    for _, row in df_subtitles.iterrows():
        ts = format_timestamp(int(row['start_frame']))
        lines.append(f"[{ts}] {row['text']}")
    return "\n".join(lines)


def sample_frames(row):
    center    = int((row['start_frame'] + row['end_frame']) // 2)
    half      = int(CONTEXT_SECONDS * FPS // 2)
    t_start   = max(0, center - half)
    t_end     = center + half
    return [t_start, center, t_end]


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are an expert basketball analyst. You will be provided with frames from a basketball game and a commentary segment.

For every request, always return a structured response with exactly these four sections:

1. DIRECT RESPONSE: What is happening in this specific moment? (1-2 sentences, factual)

2. SUMMARY: Brief summary of the player action and game context visible in the frames and commentary. Reference specific players by name where identifiable. (2-3 sentences)

3. EVIDENCE: 2-5 specific observations grounded in the commentary and visible frame content. Each piece of evidence must cite a timestamp in [MM:SS] format and indicate whether it comes from visual evidence or commentary.

4. DISCLAIMER: A brief note that this analysis is based on limited context (3 frames + one commentary segment) and may not fully reflect what actually happened in the game.

Never deviate from this four-section format. Be concise and specific."""


# =============================================================================
# DATASET GENERATION
# =============================================================================

def load_existing_dataset():
    if DATASET_PATH.exists():
        with open(DATASET_PATH, 'r') as f:
            data = json.load(f)
        print(f"Found existing dataset: {len(data)} entries")
        return data
    return {}


def save_dataset(data):
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, 'w') as f:
        json.dump(data, f, indent=2)


def generate_dataset(df_subtitles, client):
    dataset = load_existing_dataset()

    # Build full commentary transcript for context
    transcript = build_commentary_transcript(df_subtitles)

    # Filter to segments not yet processed
    pending = df_subtitles[
        ~df_subtitles.index.astype(str).isin(dataset.keys()) &
        (df_subtitles['start_frame'] >= CONTEXT_SECONDS * FPS)
    ]

    if len(pending) == 0:
        print("Dataset complete — all segments already processed.")
        return dataset

    print(f"Segments to process: {len(pending)} / {len(df_subtitles)}")

    for idx, row in tqdm(pending.iterrows(), total=len(pending), desc="Generating dataset"):
        segment_key = str(idx)

        # Sample 3 frames
        frame_indices = sample_frames(row)
        frame_paths   = [frame_path(fi) for fi in frame_indices]

        # Skip if any frame is missing
        if not all(p.exists() for p in frame_paths):
            print(f"  Skipping segment {idx} — missing frames")
            continue

        timestamp = format_timestamp(int(row['start_frame']))

        last_overload_error = None
        for attempt, delay in enumerate([0] + RETRY_DELAYS):
            if delay:
                print(f"  Overloaded — retrying in {delay}s (attempt {attempt + 1}/{len(RETRY_DELAYS) + 1})")
                time.sleep(delay)
            try:
                response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"}
                    }
                ],
                messages=[
                    # First turn: load full transcript (cached)
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Here is the full game commentary transcript:\n\n{transcript}",
                                "cache_control": {"type": "ephemeral"}
                            }
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": "Understood. I have read the full commentary transcript and am ready to analyze specific moments."
                    },
                    # Per-segment call: 3 frames + subtitle
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_image(frame_paths[0])
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_image(frame_paths[1])
                                }
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encode_image(frame_paths[2])
                                }
                            },
                            {
                                "type": "text",
                                "text": f"Commentary segment at [{timestamp}]: \"{row['text']}\"\n\nPlease analyze this moment."
                            }
                        ]
                    }
                ]
            )

                dataset[segment_key] = {
                    'idx':          idx,
                    'subtitle':     row['text'],
                    'timestamp':    timestamp,
                    'start_frame':  int(row['start_frame']),
                    'end_frame':    int(row['end_frame']),
                    'frame_paths':  [str(p) for p in frame_paths],
                    'response':     response.content[0].text
                }

                # Save after every entry — resume safely if interrupted
                save_dataset(dataset)
                break  # success — stop retrying

            except anthropic.APIStatusError as e:
                if e.status_code == 529:
                    last_overload_error = e
                    continue  # retry after delay
                print(f"  Error on segment {idx}: {e}")
                break

            except Exception as e:
                print(f"  Error on segment {idx}: {e}")
                break

        else:
            raise RuntimeError(
                f"API overloaded after all retries (10s, 20s, 40s, 1min, 5min). "
                f"Last error on segment {idx}: {last_overload_error}"
            )

    print(f"\nDataset complete: {len(dataset)} entries → {DATASET_PATH}")
    return dataset


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df_subtitles = pd.read_csv(SUBTITLES_CSV)
    client       = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    generate_dataset(df_subtitles, client)
