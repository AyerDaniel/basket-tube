'''
    https://universe.roboflow.com/basketball-stat-tracker/basketball-hoop-ball-and-player
'''

import os
import json

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor, BitsAndBytesConfig
from pathlib import Path
import pandas as pd
from PIL import Image
from finetune_vlm import concat_frames_horizontal, BasketTubeDataset

# =============================================================================
# CONFIG
# =============================================================================
PROJECT_DIR = "/home/johnsmith/Desktop/njit/workspaces/basket-tube"
CONTEXT_SECONDS  = 5
TOP_K            = 5
MODEL_DIR        = Path(f"{PROJECT_DIR}/my_basket_tube/models/context_{CONTEXT_SECONDS}s")
PALIGEMMA_DIR    = Path(f"{PROJECT_DIR}/my_basket_tube/models/paligemma_lora")
RESULTS_DIR      = Path(f"{PROJECT_DIR}/my_basket_tube/videos/query_results/context_{CONTEXT_SECONDS}s")
SUBTITLES_CSV    = f"{PROJECT_DIR}/my_basket_tube/csv/subtitles.csv"
VIDEO_PATH       = f"{PROJECT_DIR}/my_basket_tube/videos/game_1/game_1.mp4"
CACHE_PATH       = Path(f"{PROJECT_DIR}/my_basket_tube/csv/query_cache_vlm.json")
FRAMES_DIR       = Path(f"{PROJECT_DIR}/my_basket_tube/videos/game_1/frames")
MAX_NEW_TOKENS   = 512

# =============================================================================
# CONSTANTS
# =============================================================================
CLIP_MODEL = "openai/clip-vit-large-patch14"
FPS        = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# CACHE
# =============================================================================

def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH, 'r') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


# =============================================================================
# MODELS
# =============================================================================

def load_clip():
    print("Loading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).half().to(DEVICE)
    return clip_processor, clip_model


def load_index():
    index_path = MODEL_DIR / "embedding_index.pt"
    if not index_path.exists():
        raise FileNotFoundError(f"No index found at {index_path}. Run vlm_variable_context_window.py first.")
    return torch.load(index_path, weights_only=False)


def load_paligemma():
    print("Loading PaliGemma...")
    adapter_path = PALIGEMMA_DIR / "adapter_model"
    if not adapter_path.exists():
        raise FileNotFoundError(f"No adapter found at {adapter_path}. Run finetune_vlm.py first.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    pg_processor = PaliGemmaProcessor.from_pretrained(str(PALIGEMMA_DIR / "processor"))
    pg_model     = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma-3b-pt-224",
        quantization_config=bnb_config
    )
    pg_model.load_adapter(str(adapter_path))
    pg_model.eval()

    return pg_model, pg_processor


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve(query, index_data, clip_model, clip_processor):
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)

    with torch.no_grad():
        query_embedding = clip_model.text_model(**inputs).pooler_output.float()

    query_embedding = nn.functional.normalize(query_embedding, dim=-1)
    similarities    = (index_data['embeddings'].to(DEVICE) @ query_embedding.T).squeeze()
    top_indices     = similarities.topk(TOP_K).indices.cpu().numpy()

    results = []
    for idx in top_indices:
        start_frame = int(index_data['start_frames'][idx])
        seconds     = start_frame / FPS
        results.append({
            'similarity':  similarities[idx].item(),
            'start_frame': start_frame,
            'end_frame':   int(index_data['end_frames'][idx]),
            'subtitle':    str(index_data['subtitles'][idx]),
            'timestamp':   f"{int(seconds // 60)}:{int(seconds % 60):02d}"
        })

    return results


# =============================================================================
# SYNTHESIS
# =============================================================================

def frame_path(frame_idx):
    return FRAMES_DIR / f"{frame_idx}.jpg"


def sample_frames(result):
    center   = int((result['start_frame'] + result['end_frame']) // 2)
    half     = int(CONTEXT_SECONDS * FPS // 2)
    t_start  = max(0, center - half)
    t_end    = center + half
    return [t_start, center, t_end]


def synthesize(query, results, pg_model, pg_processor):
    answers = []

    for result in results:
        frame_indices = sample_frames(result)
        frame_paths   = [frame_path(fi) for fi in frame_indices]
    
        if not all(p.exists() for p in frame_paths):
            continue

        image  = concat_frames_horizontal([str(p) for p in frame_paths])
        prompt = (
            f"Commentary segment at [{result['timestamp']}]: \"{result['subtitle']}\"\n\n"
            f"Question: {query}\n\nPlease analyze this moment."
        )

        inputs = pg_processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        inputs = {k: v.to('cuda') for k, v in inputs.items()}


        with torch.no_grad():
            output_ids = pg_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        input_len = inputs['input_ids'].shape[1]
        generated = output_ids[0][input_len:]
        answer    = pg_processor.decode(generated, skip_special_tokens=True)

        answers.append({
            'timestamp': result['timestamp'],
            'subtitle':  result['subtitle'],
            'answer':    answer
        })

    return answers


# =============================================================================
# PLAYLIST
# =============================================================================

def create_playlist(query, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_query = query.replace(" ", "_").replace("'", "")[:40]
    out_path   = RESULTS_DIR / f"{safe_query}_vlm.m3u"

    with open(out_path, 'w') as f:
        f.write("#EXTM3U\n")
        for result in results:
            start_sec = max(0, result['start_frame'] / FPS - CONTEXT_SECONDS)
            end_sec   = result['start_frame'] / FPS + CONTEXT_SECONDS
            duration  = end_sec - start_sec
            label     = f"{result['timestamp']} | sim={result['similarity']:.3f} | {result['subtitle']}"

            f.write(f"#EXTINF:{duration:.1f},{label}\n")
            f.write(f"#EXTVLCOPT:start-time={start_sec:.3f}\n")
            f.write(f"#EXTVLCOPT:stop-time={end_sec:.3f}\n")
            f.write(f"{VIDEO_PATH}\n")

    return out_path


# =============================================================================
# MAIN QUERY FUNCTION
# =============================================================================

def query(user_query, clip_processor, clip_model, index_data, pg_model, pg_processor, cache):
    cache_key = user_query.lower().strip()

    if cache_key in cache:
        print("\n[cached response]\n")
        for item in cache[cache_key]['answers']:
            print(f"[{item['timestamp']}] {item['subtitle']}")
            print(item['answer'])
            print()
        playlist_path = create_playlist(user_query, cache[cache_key]['results'])
        print(f"\nPlaylist: vlc '{playlist_path}'")
        return

    results = retrieve(user_query, index_data, clip_model, clip_processor)
    answers = synthesize(user_query, results, pg_model, pg_processor)

    cache[cache_key] = {'answers': answers, 'results': results}
    save_cache(cache)

    for item in answers:
        print(f"\n[{item['timestamp']}] {item['subtitle']}")
        print(item['answer'])

    playlist_path = create_playlist(user_query, results)
    print(f"\nPlaylist: vlc '{playlist_path}'")


# =============================================================================
# UI
# =============================================================================

if __name__ == "__main__":
    clip_processor, clip_model = load_clip()
    index_data                 = load_index()
    pg_model, pg_processor     = load_paligemma()
    cache                      = load_cache()

    print("BasketTube — Player Performance Assistant (Local VLM)")
    print(f"Context window: {CONTEXT_SECONDS}s | Model: PaliGemma-3B (LoRA)")
    print("Type 'quit' to exit\n")

    while True:
        user_query = input("Query: ").strip()
        if user_query.lower() == 'quit':
            break
        if user_query:
            query(user_query, clip_processor, clip_model, index_data, pg_model, pg_processor, cache)
