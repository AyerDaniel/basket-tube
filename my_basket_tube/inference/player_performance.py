import os
import json
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path
import pandas as pd
import anthropic

# =============================================================================
# CONFIG
# =============================================================================
CONTEXT_SECONDS  = 5                  # must match the trained model you want to use
TOP_K            = 5                  # number of subtitle chunks to retrieve
MODEL_DIR        = Path(f"my_basket_tube/models/context_{CONTEXT_SECONDS}s")
RESULTS_DIR      = Path(f"my_basket_tube/videos/query_results/context_{CONTEXT_SECONDS}s")
SUBTITLES_CSV    = "my_basket_tube/csv/subtitles.csv"
VIDEO_PATH       = "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/videos/game_1/game_1.mp4"
CACHE_PATH       = Path("my_basket_tube/csv/query_cache.json")
CLAUDE_MODEL     = "claude-sonnet-4-6"

# =============================================================================
# CONSTANTS
# =============================================================================
CLIP_MODEL = "openai/clip-vit-large-patch14"
FPS        = 30
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SYSTEM_PROMPT = """You are a basketball analyst assistant. You answer questions about player performance using only the provided commentary transcript excerpts as evidence.

For each answer provide:
1. A direct response to the question
2. A short summary of the player's performance
3. 2 to 5 commentary-based pieces of evidence with timestamps
4. A brief note that the answer is based only on commentary and may not fully reflect what actually happened

Clearly separate directly stated facts from higher-level interpretations.
Always cite timestamps in [MM:SS] format.
Be concise and structured."""


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
        raise FileNotFoundError(f"No index found at {index_path}. Run sweep.py first.")
    return torch.load(index_path, weights_only=False)


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

def synthesize(query, results, client):
    chunks = "\n".join([
        f"[{r['timestamp']}] {r['subtitle']}"
        for r in results
    ])

    user_message = f"""Commentary excerpts:
{chunks}

Question: {query}"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_message}
        ]
    )

    return response.content[0].text


# =============================================================================
# PLAYLIST
# =============================================================================

def create_playlist(query, results):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_query = query.replace(" ", "_").replace("'", "")[:40]
    out_path   = RESULTS_DIR / f"{safe_query}.m3u"

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

def query(user_query, clip_processor, clip_model, index_data, client, cache):
    cache_key = user_query.lower().strip()

    # Check local cache
    if cache_key in cache:
        print("\n[cached response]\n")
        print(cache[cache_key]['answer'])
        playlist_path = create_playlist(user_query, cache[cache_key]['results'])
        print(f"\nPlaylist: vlc '{playlist_path}'")
        return

    # Retrieve
    results = retrieve(user_query, index_data, clip_model, clip_processor)

    # Synthesize
    answer = synthesize(user_query, results, client)

    # Cache
    cache[cache_key] = {'answer': answer, 'results': results}
    save_cache(cache)

    # Output
    print(f"\n{answer}")

    playlist_path = create_playlist(user_query, results)
    print(f"\nPlaylist: vlc '{playlist_path}'")


# =============================================================================
# UI
# =============================================================================

if __name__ == "__main__":
    clip_processor, clip_model = load_clip()
    index_data                 = load_index()
    client                     = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    cache                      = load_cache()

    print("BasketTube — Player Performance Assistant")
    print(f"Context window: {CONTEXT_SECONDS}s | Model: {CLAUDE_MODEL}")
    print("Type 'quit' to exit\n")

    while True:
        user_query = input("Query: ").strip()
        if user_query.lower() == 'quit':
            break
        if user_query:
            query(user_query, clip_processor, clip_model, index_data, client, cache)
