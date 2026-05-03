import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPModel, CLIPProcessor, get_linear_schedule_with_warmup
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Import Dependencies
import umap  
import numpy as np
import cv2  

# --- Config ---
CLIP_MODEL  = "openai/clip-vit-large-patch14"
EMBED_DIM  = 1024  # CLIP vision encoder output                                                                                                                                             
TEXT_DIM   = 768   # CLIP text encoder output                                                                                                                                               
N_FRAMES    = 3
PROJ_INPUT  = EMBED_DIM * N_FRAMES  # 3072
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# --- Projection Layer ---
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


# --- VLM ---
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
            cls_tokens.append(outputs.pooler_output.float())  # (B, 1024)
        combined = torch.cat(cls_tokens, dim=-1)              # (B, 3072)
        return self.projection(combined)                       # (B, 768)


# --- Dataset ---
class BasketTubeDataset(Dataset):
    def __init__(self, df_subtitles, frames_dir, clip_processor, fps=30):
        self.frames_dir = Path(frames_dir)
        self.processor  = clip_processor
        self.fps        = fps

        self.df = df_subtitles[
            df_subtitles['start_frame'] >= fps
        ].reset_index(drop=True)

        print(f"Dataset: {len(self.df)} subtitle segments")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        start = int(row['start_frame'])
        end   = int(row['end_frame'])
        mid   = (start + end) // 2

        images = []
        for fi in [start, mid, end]:
            path = self.frames_dir / f"frame_{fi + 1:04d}.jpg"
            if not path.exists():
                path = self.frames_dir / f"frame_{start + 1:04d}.jpg"
            img = Image.open(path).convert("RGB")
            images.append(img)

        pixel_values_list = [
            self.processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
            for img in images
        ]

        return {
            'pixel_values_list': pixel_values_list,
            'subtitle': row['text']
        }


# --- CLIP text embeddings ---
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
    return text_features  # (B, 1024)


# --- Load models ---
def load_models():
    print("Loading CLIP...")
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
    clip_model     = CLIPModel.from_pretrained(CLIP_MODEL).half().to(DEVICE)
    vlm            = BasketTubeVLM().to(DEVICE)
    return clip_processor, clip_model, vlm


# --- Sanity check ---
def sanity_check(df_subtitles, frames_dir, n_samples=4):
    clip_processor, clip_model, vlm = load_models()

    # Load trained projection                                                                                                                                                             
    weights_path = Path("my_basket_tube/models/projection.pt")                                                                                                                              
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

    print("\n--- Encoding frames ---")
    with torch.no_grad():
        visual_embedding = vlm.encode_frames(pixel_values_list)
    print(f"  Visual embedding shape: {visual_embedding.shape}")  # (4, 1024)

    print("\n--- Embedding subtitles via CLIP text encoder ---")
    with torch.no_grad():
        text_embedding = embed_subtitles(subtitles, clip_model, clip_processor)
    print(f"  Text embedding shape: {text_embedding.shape}")      # (4, 1024)

    print("\n--- Cosine similarity (before training) ---")
    cos = nn.CosineSimilarity(dim=-1)
    sim = cos(visual_embedding, text_embedding)
    for i, s in enumerate(sim):
        print(f"  [{i}] {s.item():.4f}")

    print("\nSanity check passed.")


# --- Training ---
def train(df_subtitles, frames_dir, epochs=3, batch_size=8, lr=1e-4):
    clip_processor, clip_model, vlm = load_models()

    # Resume from checkpoint if available                                                                                                                                                   
    weights_path = Path("my_basket_tube/models/projection.pt")
    if weights_path.exists():                                                                                                                                                               
        vlm.projection.load_state_dict(torch.load(weights_path))
        print("Resuming from checkpoint.")

    dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(vlm.projection.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=100,
        num_training_steps=len(dataloader) * epochs
    )
    cos_loss = nn.CosineEmbeddingLoss()

    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]
            subtitles         = batch['subtitle']

            visual_embedding = vlm.encode_frames(pixel_values_list)          # (B, 1024)

            with torch.no_grad():
                text_embedding = embed_subtitles(subtitles, clip_model, clip_processor)  # (B, 1024)

            targets = torch.ones(visual_embedding.shape[0]).to(DEVICE)
            loss    = cos_loss(visual_embedding, text_embedding, targets)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} loss: {total_loss / len(dataloader):.4f}")

    Path("my_basket_tube/models").mkdir(parents=True, exist_ok=True)
    torch.save(vlm.projection.state_dict(), "my_basket_tube/models/projection.pt")
    print("Saved: my_basket_tube/models/projection.pt")

                                                                                                                                                                               
                  
def run_inference_umap(df_subtitles, frames_dir):                                                                                                                                           
    clip_processor, clip_model, vlm = load_models()
                                                                                                                                                                                            
    weights_path = Path("my_basket_tube/models/projection.pt")
    vlm.projection.load_state_dict(torch.load(weights_path))
    vlm.eval()                                                                                                                                                                              
    print("Loaded trained projection weights.")
                                                                                                                                                                                            
    dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)                                                                                                                          
                                                                                                                                                                                            
    all_embeddings = []
    all_subtitles  = []                                                                                                                                                                     
                
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]                                                                                                        
            embeddings = vlm.encode_frames(pixel_values_list)                                                                                                                               
            all_embeddings.append(embeddings.cpu().numpy())                                                                                                                                 
            all_subtitles.extend(batch['subtitle'])                                                                                                                                         
                
    embeddings = np.concatenate(all_embeddings, axis=0)                                                                                                                                     
    print(f"Encoded {len(embeddings)} sequences → {embeddings.shape}")
                                                                                                                                                                                            
    print("Fitting UMAP...")                                                                                                                                                                
    reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)                                                                                                                                        
                                                                                                                                                                                            
    import plotly.graph_objects as go
    fig = go.Figure()                                                                                                                                                                       
    fig.add_trace(go.Scatter(
        x=embedding_2d[:, 0],
        y=embedding_2d[:, 1],
        mode='markers',                                                                                                                                                                     
        marker=dict(size=5, opacity=0.7),
        text=all_subtitles,                                                                                                                                                                 
        hovertemplate='%{text}<extra></extra>'
    ))
    fig.update_layout(title='BasketTube VLM — UMAP embedding', width=1200, height=800)
    fig.write_html('my_basket_tube/csv/vlm_umap.html')                                                                                                                                      
    print("Saved: vlm_umap.html")

def build_embedding_index(df_subtitles, frames_dir):                                                                                                                                        
      clip_processor, clip_model, vlm = load_models()                                                                                                                                         
                                                                                                                                                                                              
      weights_path = Path("my_basket_tube/models/projection.pt")                                                                                                                              
      vlm.projection.load_state_dict(torch.load(weights_path))                                                                                                                                
      vlm.eval()                                                                                                                                                                              
                  
      dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)                                                                                                                
      dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
                                                                                                                                                                                              
      all_embeddings = []
      all_indices    = []

      with torch.no_grad():
          for i, batch in enumerate(tqdm(dataloader, desc="Building index")):
              pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]                                                                                                        
              embeddings = vlm.encode_frames(pixel_values_list)                                                                                                                               
              all_embeddings.append(embeddings.cpu())                                                                                                                                         
              all_indices.extend(range(i * 16, i * 16 + len(embeddings)))                                                                                                                     
                                                                                                                                                                                              
      embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)                                                                                                                               
                                                                                                                                                                                              
      # Normalize for cosine similarity                                                                                                                                                       
      embeddings = nn.functional.normalize(embeddings, dim=-1)
                                                                                                                                                                                              
      # Save index
      index_data = {                                                                                                                                                                          
          'embeddings': embeddings,
          'start_frames': df_subtitles['start_frame'].values[:len(embeddings)],
          'end_frames': df_subtitles['end_frame'].values[:len(embeddings)],                                                                                                                   
          'subtitles': df_subtitles['text'].values[:len(embeddings)]
      }                                                                                                                                                                                       
      torch.save(index_data, 'my_basket_tube/models/embedding_index.pt')
      print(f"Saved index: {len(embeddings)} embeddings")                                                                                                                                     
      return index_data
                                                                                                                                                                                              
                                                                                                                                                                                              
def retrieve(query, index_data, clip_model, clip_processor, top_k=5):                                                                                                                       
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
                                                                                                                                                                                            
    similarities = (index_data['embeddings'].to(DEVICE) @ query_embedding.T).squeeze()                                                                                                      
    top_indices  = similarities.topk(top_k).indices.cpu().numpy()
                                                                                                                                                                                            
    results = []
    for idx in top_indices:                                                                                                                                                                 
        results.append({
            'similarity':   similarities[idx].item(),
            'start_frame':  index_data['start_frames'][idx],
            'end_frame':    index_data['end_frames'][idx],                                                                                                                                  
            'subtitle':     index_data['subtitles'][idx],
            'timestamp':    f"{int(index_data['start_frames'][idx] / 30 // 60)}:{int(index_data['start_frames'][idx] / 30 % 60):02d}"                                                       
        })                                                                                                                                                                                  

    return results                                                                                                                                                                          
                
                                                                                                                                                                                            
def query_video(query, frames_dir):
    clip_processor, clip_model, vlm = load_models()                                                                                                                                         
                
    index_path = Path("my_basket_tube/models/embedding_index.pt")                                                                                                                           
    if not index_path.exists():
        df_subtitles = pd.read_csv('my_basket_tube/csv/subtitles.csv')                                                                                                                      
        index_data   = build_embedding_index(df_subtitles, frames_dir)                                                                                                                      
    else:
        index_data = torch.load(index_path, weights_only=False)                                                                                                                                                 
        print("Loaded existing index.")
                                                                                                                                                                                            
    results = retrieve(query, index_data, clip_model, clip_processor)
                                                                                                                                                                                            
    print(f"\nQuery: '{query}'")
    print("-" * 60)
    for r in results:
        print(f"  [{r['timestamp']}] sim={r['similarity']:.3f} — {r['subtitle']}")                                                                                                          

    return results   
def build_embedding_index(df_subtitles, frames_dir):
      clip_processor, clip_model, vlm = load_models()

      weights_path = Path("my_basket_tube/models/projection.pt")
      vlm.projection.load_state_dict(torch.load(weights_path))
      vlm.eval()

      dataset    = BasketTubeDataset(df_subtitles, frames_dir, clip_processor)
      dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

      all_embeddings = []
      all_indices    = []

      with torch.no_grad():
          for i, batch in enumerate(tqdm(dataloader, desc="Building index")):
              pixel_values_list = [pv.to(DEVICE) for pv in batch['pixel_values_list']]
              embeddings = vlm.encode_frames(pixel_values_list)
              all_embeddings.append(embeddings.cpu())
              all_indices.extend(range(i * 16, i * 16 + len(embeddings)))

      embeddings = torch.cat(all_embeddings, dim=0)  # (N, 768)

      # Normalize for cosine similarity
      embeddings = nn.functional.normalize(embeddings, dim=-1)

      # Save index
      index_data = {
          'embeddings': embeddings,
          'start_frames': df_subtitles['start_frame'].values[:len(embeddings)],
          'end_frames': df_subtitles['end_frame'].values[:len(embeddings)],
          'subtitles': df_subtitles['text'].values[:len(embeddings)]
      }
      torch.save(index_data, 'my_basket_tube/models/embedding_index.pt')
      print(f"Saved index: {len(embeddings)} embeddings")
      return index_data


def retrieve(query, index_data, clip_model, clip_processor, top_k=5):
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

    similarities = (index_data['embeddings'].to(DEVICE) @ query_embedding.T).squeeze()
    top_indices  = similarities.topk(top_k).indices.cpu().numpy()

    results = []
    for idx in top_indices:
        results.append({
            'similarity':   similarities[idx].item(),
            'start_frame':  index_data['start_frames'][idx],
            'end_frame':    index_data['end_frames'][idx],
            'subtitle':     index_data['subtitles'][idx],
            'timestamp':    f"{int(index_data['start_frames'][idx] / 30 // 60)}:{int(index_data['start_frames'][idx] / 30 % 60):02d}"
        })

    return results


def query_video(query, frames_dir):
    clip_processor, clip_model, vlm = load_models()

    index_path = Path("my_basket_tube/models/embedding_index.pt")
    if not index_path.exists():
        df_subtitles = pd.read_csv('my_basket_tube/csv/subtitles.csv')
        index_data   = build_embedding_index(df_subtitles, frames_dir)
    else:
        index_data = torch.load(index_path, weights_only=False)
        print("Loaded existing index.")

    results = retrieve(query, index_data, clip_model, clip_processor)

    print(f"\nQuery: '{query}'")
    print("-" * 60)
    for r in results:
        print(f"  [{r['timestamp']}] sim={r['similarity']:.3f} — {r['subtitle']}")

    create_query_playlist(                                                                                                                                                                      
      query,                                                                                                                                                                                  
      results,                                                                                                                                                                                
      "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/videos/game_1/game_1.mp4"                                                                                           
  )   
    return results

def create_query_playlist(query, results, video_path, context_seconds=10):                                                                                                                   
    output_dir  = Path("my_basket_tube/videos/query_results")
    output_dir.mkdir(parents=True, exist_ok=True)                                                                                                                                           

    safe_query  = query.replace(" ", "_").replace("'", "")[:40]                                                                                                                             
    out_path    = output_dir / f"{safe_query}.m3u"
                                                                                                                                                                                            
    with open(out_path, 'w') as f:
        f.write("#EXTM3U\n")
        for result in results:                                                                                                                                                              
            start_sec = max(0, result['start_frame'] / 30 - context_seconds)
            end_sec   = result['start_frame'] / 30 + context_seconds                                                                                                                        
            duration  = end_sec - start_sec
            label     = f"{result['timestamp']} | sim={result['similarity']:.3f} | {result['subtitle']}"                                                                                    
                                                                                                                                                                                            
            f.write(f"#EXTINF:{duration:.1f},{label}\n")
            f.write(f"#EXTVLCOPT:start-time={start_sec:.3f}\n")                                                                                                                             
            f.write(f"#EXTVLCOPT:stop-time={end_sec:.3f}\n")                                                                                                                                
            f.write(f"{video_path}\n")                                                                                                                                                      
                                                                                                                                                                                            
    print(f"Saved: {out_path}")                                                                                                                                                             
    print(f"Open with: vlc {out_path}")
    return out_path


if __name__ == "__main__":
    df_subtitles = pd.read_csv('my_basket_tube/csv/subtitles.csv')
    frames_dir   = "my_basket_tube/videos/game_1/frames"
    #sanity_check(df_subtitles, frames_dir)
    #train(df_subtitles, frames_dir, epochs=6, lr=1e-5)
    #run_inference_umap(df_subtitles, frames_dir) 
    
    # Run on several queries.
    queries = []
    # queries += ["Curry hits a three pointer.","Curry","three pointer","platypus"]
    # queries += ["James"]
    queries += ["Who made the most points?"]
    
    for query in queries:
        query_video(query, "my_basket_tube/videos/game_1/frames")

