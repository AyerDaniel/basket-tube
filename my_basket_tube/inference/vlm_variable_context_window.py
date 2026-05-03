import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# =============================================================================
# CONFIG — modify these before running
# =============================================================================
CONTEXT_SECONDS = 5                  # seconds of context around each subtitle segment center
EPOCHS          = 3                  # training epochs
BATCH_SIZE      = 8                  # training batch size
LR              = 1e-4               # learning rate
TOP_K           = 5                  # number of results to return per query

SUBTITLES_CSV   = "my_basket_tube/csv/subtitles.csv"
FRAMES_DIR      = "my_basket_tube/videos/game_1/frames"
VIDEO_PATH      = "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/videos/game_1/game_1.mp4"
MODEL_DIR       = Path(f"my_basket_tube/models/context_{CONTEXT_SECONDS}s")
RESULTS_DIR     = Path(f"my_basket_tube/videos/query_results/context_{CONTEXT_SECONDS}s")

# =============================================================================
# CONSTANTS — do not modify
# =============================================================================
CLIP_MODEL  = "openai/clip-vit-large-patch14"
EMBED_DIM   = 1024   # CLIP vision encoder CLS output dim
TEXT_DIM    = 768    # CLIP text encoder output dim
N_FRAMES    = 3
PROJ_INPUT  = EMBED_DIM * N_FRAMES   # 3072
FPS         = 30
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class BasketTubeVLM(nn.Module):
    def __init__(self):
        super().__init__()

        # ViT — frozen
        self.clip = CLIPModel.from_pretrained(CLIP_MODEL).vision_model.half()
        for param in self.clip.parameters():
            param.requires_grad = False

        # Projection — trained
        self.projection = ProjectionLayer(PROJ_INPUT, TEXT_DIM)

    def encode_frames(self, pixel_values_list):
        cls_tokens = []
        for pixel_values in pixel_values_list:
            outputs = self.clip(pixel_values=pixel_values.half())
            cls_tokens.append(outputs.pooler_output.float())   # (B, 1024)
        combined = torch.cat(cls_tokens, dim=-1)               # (B, 3072)
        return self.projection(combined)                        # (B, 768)


# =============================================================================
# DATASET
# =============================================================================

class BasketTubeDataset(Dataset):
    def __init__(self, df_subtitles, frames_dir, clip_processor):
        self.frames_dir = Path(frames_dir)
        self.processor  = clip_processor
        self.half_window = int(CONTEXT_SECONDS * FPS // 2)

        self.df = df_subtitles[
            df_subtitles['start_frame'] >= CONTEXT_SECONDS * FPS
        ].reset_index(drop=True)

        print(f"Dataset: {len(self.df)} subtitle segments "
              f"(context window: {CONTEXT_SECONDS}s)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        center = int((row['start_frame'] + row['end_frame']) // 2)

        t_start = max(0, center - self.half_window)
        t_end   = center + self.half_window
        t_mid   = center

        images = []
        for fi in [t_start, t_mid, t_end]:
            path = self.frames_dir / f"frame_{fi + 1:04d}.jpg"
            if not path.exists():
                path = self.frames_dir / f"frame_{center + 1:04d}.jpg"
            img = Image.open(path).convert("RGB")
            images.append(img)

        pixel_values_list = [
            self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            for img in images
        ]

        return {
            'pixel_values_list': pixel_values_list,
            'subtitle':          row['text'],
            'start_frame':       row['start_frame'],
            'end_frame':         row['end_frame'],
        }


# =============================================================================
# HELPERS
# =============================================================================

def load_models():
    print("Loading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).half().to(DEVICE)
    vlm            = BasketTubeVLM().to(DEVICE)
    return clip_processor, clip_model, vlm


def embed_subtitles(subtitles, clip_model, clip_processor):
    inputs = clip_processor(
        text=list(subtitles),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    ).to(DEVICE)
    with torch.no_grad():
        text_features = clip_model.text_model(**inputs).pooler_output.float()
    return text_features   # (B, 768)


# =============================================================================
# TRAINING
# =============================================================================

def train(df_subtitles, frames_dir):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clip_processor, clip_model, vlm = load_models()

    weights_path = MODEL_DIR / "projection.pt"
    if weights_path.exists():
        vlm.projection.load_state_dict(torch.load(weights_path))
        print("Resuming from checkpoint.")

    dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(vlm.projection.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * EPOCHS
    )
    cos_loss = nn.CosineEmbeddingLoss()

    vlm.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]
            subtitles         = batch['subtitle']

            visual_embedding = vlm.encode_frames(pixel_values_list)

            with torch.no_grad():
                text_embedding = embed_subtitles(subtitles, clip_model, clip_processor)

            targets = torch.ones(visual_embedding.shape[0]).to(DEVICE)
            loss    = cos_loss(visual_embedding, text_embedding, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

    torch.save(vlm.projection.state_dict(), weights_path)
    print(f"Saved: {weights_path}")


# =============================================================================
# INDEXING
# =============================================================================

def build_embedding_index(df_subtitles, frames_dir):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    clip_processor, clip_model, vlm = load_models()

    weights_path = MODEL_DIR / "projection.pt"
    vlm.projection.load_state_dict(torch.load(weights_path))
    vlm.eval()

    dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_embeddings  = []
    all_start_frames = []
    all_end_frames   = []
    all_subtitles    = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Building index"):
            pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]
            embeddings = vlm.encode_frames(pixel_values_list)
            all_embeddings.append(embeddings.cpu())
            all_start_frames.extend(batch['start_frame'].tolist())
            all_end_frames.extend(batch['end_frame'].tolist())
            all_subtitles.extend(batch['subtitle'])

    embeddings = torch.cat(all_embeddings, dim=0)
    embeddings = nn.functional.normalize(embeddings, dim=-1)

    index_data = {
        'embeddings':   embeddings,
        'start_frames': np.array(all_start_frames),
        'end_frames':   np.array(all_end_frames),
        'subtitles':    np.array(all_subtitles)
    }

    index_path = MODEL_DIR / "embedding_index.pt"
    torch.save(index_data, index_path, pickle_protocol=4)
    print(f"Saved index: {len(embeddings)} embeddings → {index_path}")
    return index_data


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
        results.append({
            'similarity':  similarities[idx].item(),
            'start_frame': start_frame,
            'end_frame':   int(index_data['end_frames'][idx]),
            'subtitle':    str(index_data['subtitles'][idx]),
            'timestamp':   f"{int(start_frame / FPS // 60)}:{int(start_frame / FPS % 60):02d}"
        })

    return results


# =============================================================================
# PLAYLIST
# =============================================================================

def create_query_playlist(query, results, video_path):
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
            f.write(f"{video_path}\n")

    print(f"Saved: {out_path}")
    print(f"Open with: vlc '{out_path}'")
    return out_path


# =============================================================================
# QUERY
# =============================================================================

def query_video(query):
    clip_processor, clip_model, vlm = load_models()

    weights_path = MODEL_DIR / "projection.pt"
    index_path   = MODEL_DIR / "embedding_index.pt"

    if not weights_path.exists():
        print("No trained model found. Run train() first.")
        return

    vlm.projection.load_state_dict(torch.load(weights_path))
    vlm.eval()

    if not index_path.exists():
        print("No index found. Run build_embedding_index() first.")
        return

    index_data = torch.load(index_path, weights_only=False)

    results = retrieve(query, index_data, clip_model, clip_processor)

    print(f"\nQuery: '{query}'  [context: {CONTEXT_SECONDS}s]")
    print("-" * 60)
    for r in results:
        print(f"  [{r['timestamp']}] sim={r['similarity']:.3f} — {r['subtitle']}")

    create_query_playlist(query, results, VIDEO_PATH)
    return results


# =============================================================================
# SANITY CHECK
# =============================================================================

def sanity_check(df_subtitles, frames_dir, n_samples=4):
    clip_processor, clip_model, vlm = load_models()

    weights_path = MODEL_DIR / "projection.pt"
    if weights_path.exists():
        vlm.projection.load_state_dict(torch.load(weights_path))
        print("Loaded trained projection weights.")

    dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)
    dataloader = DataLoader(dataset, batch_size=n_samples, shuffle=True)

    batch = next(iter(dataloader))
    pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]
    subtitles = batch['subtitle']

    print("\n--- Subtitle labels ---")
    for i, s in enumerate(subtitles):
        print(f"  [{i}] {s}")

    with torch.no_grad():
        visual_embedding = vlm.encode_frames(pixel_values_list)
        text_embedding   = embed_subtitles(subtitles, clip_model, clip_processor)

    print(f"\n  Visual embedding shape: {visual_embedding.shape}")
    print(f"  Text embedding shape:   {text_embedding.shape}")

    cos = nn.CosineSimilarity(dim=-1)
    sim = cos(visual_embedding, text_embedding)
    print("\n--- Cosine similarity ---")
    for i, s in enumerate(sim):
        print(f"  [{i}] {s.item():.4f}")

    print("\nSanity check passed.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    df_subtitles = pd.read_csv(SUBTITLES_CSV)

    # Step 1: Train
    train(df_subtitles, FRAMES_DIR)

    # Step 2: Build index
    build_embedding_index(df_subtitles, FRAMES_DIR)

    # Step 3: Query
    query_video("Curry hits a three pointer")

    
