'''
Reference material:  https://www.youtube.com/watch?v=yGQb9KkvQ1Q
Using detection model rf-detr: https://github.com/roboflow/rf-detr/blob/develop/README.md

'''

'''
    I am creating this with a different pipeline flow in mind.
    1)  Cut video into 1 second frames.
    2)  Identify semantic events purely based on ball motion. (i.e. Dribble, Pass, Shoot, etc).
    3)  Attribute actor to actions only at bookends of semantic events.


        


    -----------------
    Expected Detected Classes Are:
    1: 'person'                                                                                                                                                                                 
    2: 'bicycle'                                                                                                                                                                                
    3: 'car'                                                                                                                                                                                    
    4: 'motorcycle'
    5: 'airplane'                                                                                                                                                                               
    6: 'bus'        
    7: 'train'
    8: 'truck'
    9: 'boat'
    10: 'traffic light'
    11: 'fire hydrant'                                                                                                                                                                          
    13: 'stop sign'
    14: 'parking meter'                                                                                                                                                                         
    15: 'bench'     
    16: 'bird'
    17: 'cat'
    18: 'dog'
    19: 'horse'
    20: 'sheep'                                                                                                                                                                                 
    21: 'cow'
    22: 'elephant'                                                                                                                                                                              
    23: 'bear'      
    24: 'zebra'
    25: 'giraffe'
    27: 'backpack'
    28: 'umbrella'
    31: 'handbag'
    32: 'tie'                                                                                                                                                                                   
    33: 'suitcase'
    34: 'frisbee'                                                                                                                                                                               
    35: 'skis'      
    36: 'snowboard'
    37: 'sports ball'
    38: 'kite'
    39: 'baseball bat'
    40: 'baseball glove'                                                                                                                                                                        
    41: 'skateboard'
    42: 'surfboard'                                                                                                                                                                             
    43: 'tennis racket'
    44: 'bottle'
    46: 'wine glass'
    47: 'cup'
    48: 'fork'
    49: 'knife'                                                                                                                                                                                 
    50: 'spoon'
    51: 'bowl'                                                                                                                                                                                  
    52: 'banana'    
    53: 'apple'
    54: 'sandwich'
    55: 'orange'
    56: 'broccoli'
    57: 'carrot'                                                                                                                                                                                
    58: 'hot dog'
    59: 'pizza'                                                                                                                                                                                 
    60: 'donut'     
    61: 'cake'
    62: 'chair'
    63: 'couch'
    64: 'potted plant'
    65: 'bed'
    67: 'dining table'                                                                                                                                                                          
    70: 'toilet'
    72: 'tv'                                                                                                                                                                                    
    73: 'laptop'    
    74: 'mouse'
    75: 'remote'
    76: 'keyboard'
    77: 'cell phone'
    78: 'microwave'                                                                                                                                                                             
    79: 'oven'
    80: 'toaster'                                                                                                                                                                               
    81: 'sink'      
    82: 'refrigerator'
    84: 'book'
    85: 'clock'
    86: 'vase'
    87: 'scissors'
    88: 'teddy bear'                                                                                                                                                                            
    89: 'hair drier'
    90: 'toothbrush'  

'''

# Import dependencies.
import pandas as pd                                                                                                                                                                   
import numpy as np   
import matplotlib
# matplotlib.use("Agg")

import matplotlib.pyplot as plt
import json

from scipy.signal import find_peaks
from scipy.signal import stft
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import torch                                                                                                                                                                                
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F 

from rfdetr import RFDETRLarge
from rfdetr.assets.coco_classes import COCO_CLASSES
import os

from sklearn.cluster import KMeans                                                                                                                                                      
import umap 

# Import progress bar.
from tqdm import tqdm

import yt_dlp  

import subprocess
from pathlib import Path 

import cv2
from dtaidistance import dtw_ndim   
import plotly.graph_objects as go  
import webvtt   

### Set video working directory.
video_dir = "my_basket_tube/videos"

### Refactor List ###
'''
    Check for dependencies and install if absent.
    Right now this just grabs this specific video. 
    Expand it out to prompt user for which video to digest.
    Add filename feature for if_exists check.
'''

### Classes ###

class BallTrajectoryDataset(Dataset):                                                                                                                                                       
    def __init__(self, sequences, num_frames=20):
        self.sequences = sequences                                                                                                                                                          
        self.num_frames = num_frames
                                                                                                                                                                                            
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]['sequence']
                                                                                                                                                                                            
        # Sample num_frames evenly from sequence regardless of length
        indices = np.linspace(0, len(seq) - 1, self.num_frames, dtype=int)                                                                                                                  
        sampled = seq[indices]  # (num_frames, 4)                                                                                                                                           
                                                                                                                                                                                            
        return torch.tensor(sampled, dtype=torch.float32)
      

      
### Functions ###


def assign_frame_labels(df, sequences, labels):
    label_map = {seq['center_frame']: int(label)
                for seq, label in zip(sequences, labels)}                                                                                                                                  
    df['cluster'] = df['frame_idx'].map(label_map)
    return df 

# End of assign_frame_labels().

def estimate_camera_motion(frame1, frame2, grid_density=20):                                                                                                                                
      gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
      gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)                                                                                                                                        
                                                                                                                                                                                              
      h, w = gray1.shape
      xs = np.linspace(50, w - 50, grid_density).astype(np.float32)                                                                                                                           
      ys = np.linspace(50, h - 50, grid_density).astype(np.float32)
      grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32).reshape(-1, 1, 2)                                                                                                   
   
      tracked, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, grid, None)                                                                                                                 
                  
      good_src = grid[status.flatten() == 1]                                                                                                                                                  
      good_dst = tracked[status.flatten() == 1]

      if len(good_src) < 4:                                                                                                                                                                   
          return 0.0, 0.0
                                                                                                                                                                                              
      flow = good_dst - good_src
      cam_dx = float(np.median(flow[:, 0, 0]))
      cam_dy = float(np.median(flow[:, 0, 1]))
                                                                                                                                                                                              
      return cam_dx, cam_dy

def compute_motion_vectors(df, frames_dir=None):
    df = df.copy()         
    df['cx'] = (df['x1'] + df['x2']) / 2                                                                                                                                                    
    df['cy'] = (df['y1'] + df['y2']) / 2
    df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])                                                                                                                              
                                                                                                                                                                                            
    df['raw_dx'] = df['cx'].diff()                                                                                                                                                          
    df['raw_dy'] = df['cy'].diff()                                                                                                                                                          
    df['d_area'] = df['area'].diff()                                                                                                                                                        

    # Camera motion correction via background optical flow                                                                                                                                  
    if frames_dir is not None:
        frames_dir = Path(frames_dir)                                                                                                                                                       
        cam_dx_list = [np.nan]
        cam_dy_list = [np.nan]                                                                                                                                                              
                                                                                                                                                                                            
        frame_indices = df['frame_idx'].values
        for i in range(1, len(frame_indices)):                                                                                                                                              
            f0 = frames_dir / f"frame_{frame_indices[i-1] + 1:04d}.jpg"
            f1 = frames_dir / f"frame_{frame_indices[i] + 1:04d}.jpg"                                                                                                                       

            img0 = cv2.imread(str(f0))                                                                                                                                                      
            img1 = cv2.imread(str(f1))
                                                                                                                                                                                            
            if img0 is None or img1 is None:
                cam_dx_list.append(0.0)                                                                                                                                                     
                cam_dy_list.append(0.0)
                continue

            cam_dx, cam_dy = estimate_camera_motion(img0, img1)                                                                                                                             
            cam_dx_list.append(cam_dx)
            cam_dy_list.append(cam_dy)                                                                                                                                                      
                
        df['cam_dx'] = cam_dx_list
        df['cam_dy'] = cam_dy_list
        df['corrected_dx'] = df['raw_dx'] - df['cam_dx']                                                                                                                                    
        df['corrected_dy'] = df['raw_dy'] - df['cam_dy']
    else:                                                                                                                                                                                   
        df['cam_dx'] = 0.0
        df['cam_dy'] = 0.0                                                                                                                                                                  
        df['corrected_dx'] = df['raw_dx']
        df['corrected_dy'] = df['raw_dy']

    # Normalize by ball size to control for zoom
    df['dx'] = df['corrected_dx'] / np.sqrt(df['area'])
    df['dy'] = df['corrected_dy'] / np.sqrt(df['area'])                                                                                                                                     

    return df.dropna().reset_index(drop=True)  

# End of compute_motion_vectors().

def detect_ball():
    # Function to identify in which frames the ball occurs.

    # Get sorted list of frames. 
    frames_dir = f"{video_dir}/game_1/frames"                                                                                                                                                   
    frames = sorted([f"{frames_dir}/{f}" for f in os.listdir(frames_dir)])   

    # Create directory for frames containing ball.
    ball_frames_dir = Path(f"{video_dir}/game_1/ball_frames")
    ball_frames_dir.mkdir(parents=True, exist_ok=True)   

    # Create list to write to dataframe.
    rows = []

    # My experimental pipeline:
    balls =[]
    
    # Batching 16 images on RTX 3080 with 16GB VRAM, CUDA 13.0, driver 580.95.05.
    batch_size = 16

    # Import model.
    model = RFDETRLarge()
    model.optimize_for_inference(batch_size=batch_size)

    # Batch detect on images.
    for batch_start in tqdm(range(0, len(frames), batch_size)):

        # Batch frames.                                                                                                                                       
        batch = frames[batch_start : batch_start + batch_size]

        #Last batch fails if not of size batch_size.
        original_len = len(batch)                                                                                                                                                                   
        if original_len < batch_size:                                                                                                                                                               
            batch = batch + [batch[-1]] * (batch_size - original_len)
        

        # Run batch detections.
        batch_detections = model.predict(batch, threshold=0.5)      

        # If there are more detections they are dummies used to padd the last batch.  Grab all entries that are genuine.
        batch_detections = batch_detections[:original_len]
                                                                                                        
        # Extract detections.                                                                                                                           
        for i, detections in enumerate(batch_detections):     

            # Model was trained on COCO.  We're looking for classes: 1: person and 37: sports ball

            # Set frame_idx.
            frame_idx = batch_start + i   

            # Iterate detections.
            for j in range(len(detections)):     

                # We're loking for the ball only.
                if detections.class_id[j] == 37:

                    x1         = int(detections.xyxy[j][0])                                                                                                                                             
                    y1         = int(detections.xyxy[j][1])
                    x2         = int(detections.xyxy[j][2])                                                                                                                                             
                    y2         = int(detections.xyxy[j][3])
                    confidence = detections.confidence[j]                                                                                                                                               
                    class_id   = detections.class_id[j]                                                                                                                                                 
                    class_name = COCO_CLASSES.get(class_id, str(class_id))
                    
                    # Append to list.                                                                                                                                              
                    balls.append({                                                                                                                                                                   
                        'frame_idx': frame_idx,                                                                                                                                                         
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'confidence': confidence,
                        'class_id': class_id,                                                                                                                                                           
                        'class_name': class_name
                    })

                    # Create annotation copy of frame and store.
                    frame_path = frames[batch_start + i]                                                                                                                                                
                    img = cv2.imread(frame_path)
                    if img is not None:
                        overlay = img.copy()                                                                                                                                                            
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), -1)
                        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)                                                                                                                                 
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)                                                                                                                        
                        cv2.putText(img, 'ball', (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)                                                                                                                    
                        out_path = ball_frames_dir / f"frame_{frame_idx}.jpg"                                                                                                                   
                        cv2.imwrite(str(out_path), img)
    

            # Standard Pipeline:
            frame_idx = batch_start + i   

            for j in range(len(detections)):                                                                                                                                                    
                rows.append({                                                                                                                                                                   
                    'frame_idx': frame_idx,
                    'x1': detections.xyxy[j][0],
                    'y1': detections.xyxy[j][1],
                    'x2': detections.xyxy[j][2],
                    'y2': detections.xyxy[j][3],
                    'confidence': detections.confidence[j],
                    'class_id': detections.class_id[j],
                    'class_name': COCO_CLASSES.get(detections.class_id[j], str(detections.class_id[j])),
                    #'source_image' : detections.data['source_image'][j] # This stores an enitre freakin image.  HUGE!
                })

    # Write rows to dataframe.
    df_detections = pd.DataFrame(rows)

    # Experimental Pipeline:
    df_balls = pd.DataFrame(balls)

    # Store in file for now.
    df_detections.to_csv('my_basket_tube/csv/df_detections.csv')

    # Experimental Pipeline:
    df_balls.to_csv('my_basket_tube/csv/df_balls.csv')

# End of detect_ball()

def detect_players():  
                                                                                                                                                                           
    ball_frames_dir      = Path(f"{video_dir}/game_1/ball_frames")
    ball_player_frames_dir = Path(f"{video_dir}/game_1/ball_player_frames")                                                                                                                 
    ball_player_frames_dir.mkdir(parents=True, exist_ok=True)                                                                                                                               
                                                                                                                                                                                            
    # Load ball centroids for nearest-player calculation                                                                                                                                    
    df_balls = pd.read_csv('my_basket_tube/csv/df_balls.csv', index_col=0)                                                                                                                  
    df_balls = df_balls.sort_values('confidence', ascending=False).drop_duplicates('frame_idx').sort_values('frame_idx')                                                                    
    df_balls['cx'] = ((df_balls['x1'] + df_balls['x2']) / 2).astype(int)                                                                                                                    
    df_balls['cy'] = ((df_balls['y1'] + df_balls['y2']) / 2).astype(int)                                                                                                                    
    ball_centroid_map = dict(zip(df_balls['frame_idx'], zip(df_balls['cx'], df_balls['cy'])))                                                                                               
                                                                                                                                                                                            
    frame_files = sorted(ball_frames_dir.glob("frame_*.jpg"))                                                                                                                               
    frames = [str(f) for f in frame_files]                                                                                                                                                  

    batch_size = 16
                                                                                                                                                                            
    model = RFDETRLarge()                                                                                                                                                                   
    model.optimize_for_inference(batch_size=batch_size)
                                                                                                                                                                                            
    rows = []   


    for batch_start in tqdm(range(0, len(frames), batch_size)):                                                                                                                             
        batch = frames[batch_start : batch_start + batch_size]
        original_len = len(batch)                                                                                                                                                           
        if original_len < batch_size:
            batch = batch + [batch[-1]] * (batch_size - original_len)
                                                                                                                                                                                            
        batch_detections = model.predict(batch, threshold=0.5)
        batch_detections = batch_detections[:original_len]                                                                                                                                  
                
        for i, detections in enumerate(batch_detections):
            frame_idx = int(Path(frames[batch_start + i]).stem.split('_')[1]) - 1
                                                                                                                                                                                            
            # Collect all person detections for this frame                                                                                                                                  
            players = []                                                                                                                                                                    
            for j in range(len(detections)):                                                                                                                                                
                if detections.class_id[j] == 1:
                    x1 = int(detections.xyxy[j][0])
                    y1 = int(detections.xyxy[j][1])                                                                                                                                         
                    x2 = int(detections.xyxy[j][2])
                    y2 = int(detections.xyxy[j][3])                                                                                                                                         
                    pcx = (x1 + x2) // 2                                                                                                                                                    
                    pcy = (y1 + y2) // 2
                    players.append({                                                                                                                                                        
                        'frame_idx': frame_idx,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,                                                                                                                             
                        'cx': pcx, 'cy': pcy,
                        'confidence': detections.confidence[j],                                                                                                                             
                    })
                                                                                                                                                                                            
            # Find nearest player to ball
            nearest_idx = None
            if players and frame_idx in ball_centroid_map:
                bcx, bcy = ball_centroid_map[frame_idx]                                                                                                                                     
                dists = [np.sqrt((p['cx'] - bcx)**2 + (p['cy'] - bcy)**2) for p in players]
                nearest_idx = int(np.argmin(dists))                                                                                                                                         
                for k, p in enumerate(players):                                                                                                                                             
                    p['nearest_to_ball'] = (k == nearest_idx)                                                                                                                               
                    p['dist_to_ball'] = dists[k]                                                                                                                                            
            else:
                for p in players:                                                                                                                                                           
                    p['nearest_to_ball'] = False
                    p['dist_to_ball'] = None                                                                                                                                                

            rows.extend(players)                                                                                                                                                            
                
            # Annotate and save frame                                                                                                                                                       
            img = cv2.imread(frames[batch_start + i])
            if img is None:                                                                                                                                                                 
                continue

            for k, p in enumerate(players):                                                                                                                                                 
                color = (0, 165, 255) if k == nearest_idx else (255, 255, 255)
                cv2.rectangle(img, (p['x1'], p['y1']), (p['x2'], p['y2']), color, 2)                                                                                                        
                label = 'nearest' if k == nearest_idx else 'player'                                                                                                                         
                cv2.putText(img, label, (p['x1'], p['y1'] - 5),                                                                                                                             
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)                                                                                                                        
                                                                                                                                                                                            
            out_path = ball_player_frames_dir / Path(frames[batch_start + i]).name                                                                                                          
            cv2.imwrite(str(out_path), img)
                                                                                                                                                                                            
    df_players = pd.DataFrame(rows)                                                                                                                                                         
    df_players.to_csv('my_basket_tube/csv/df_players.csv')
    print(f"Saved: df_players.csv  ({len(df_players)} player detections across {len(frames)} frames)")

# End of detect_players().

def parse_subtitles():
    vtt_path = f'{video_dir}/game_1/game_1.en.vtt'                                                                                                                                          
    rows = []   
    for caption in webvtt.read(vtt_path):                                                                                                                                                   
        rows.append({
            'start': caption.start,                                                                                                                                                         
            'end': caption.end,
            'start_seconds': caption.start_in_seconds,
            'end_seconds': caption.end_in_seconds,                                                                                                                                          
            'text': caption.text.strip()
        })                                                                                                                                                                                  
    df = pd.DataFrame(rows)
    df.to_csv('my_basket_tube/csv/subtitles.csv')                                                                                                                                           
    print(f"Parsed {len(df)} subtitle entries")                                                                                                                                             
    return df

def parse_live_chat():
    chat_path = f'{video_dir}/game_1/game_1.live_chat.json'
    rows = []                                                                                                                                                                               

    with open(chat_path, 'r') as f:                                                                                                                                                         
        for line in f:
            try:                                                                                                                                                                            
                obj = json.loads(line.strip())
                offset_ms = int(obj.get('replayChatItemAction', {}).get('videoOffsetTimeMsec', -1))
                if offset_ms < 0:                                                                                                                                                           
                    continue
                                                                                                                                                                                            
                actions = obj.get('replayChatItemAction', {}).get('actions', [])                                                                                                            
                for action in actions:
                    renderer = action.get('addChatItemAction', {}).get('item', {}).get('liveChatTextMessageRenderer', {})                                                                   
                    if not renderer:                                                                                                                                                        
                        continue
                    runs = renderer.get('message', {}).get('runs', [])                                                                                                                      
                    text = ' '.join(r.get('text', '') for r in runs).strip()                                                                                                                
                    if text:
                        rows.append({                                                                                                                                                       
                            'offset_ms': offset_ms,
                            'offset_sec': offset_ms / 1000,                                                                                                                                 
                            'frame_idx': int((offset_ms / 1000) * 30),
                            'text': text                                                                                                                                                    
                        })
            except:                                                                                                                                                                         
                continue
                                                                                                                                                                                            
    df = pd.DataFrame(rows).sort_values('offset_ms').reset_index(drop=True)
    df.to_csv('my_basket_tube/csv/live_chat.csv')                                                                                                                                           
    print(f"Parsed {len(df)} chat messages")                                                                                                                                                
    return df

def import_video():
    # Function to download the video.                                                                                                                                                                            

    # Define format and output filename.                                                                                            
    ydl_opts = {                                                                                                                                                                 
        'writesubtitles': True,
        'writeautomaticsub': True,                                                                                                                                                          
        'subtitleslangs': ['live_chat'], # This is the option from this video specifically.  Rework for a generalized flow.
        'subtitlesformat': 'vtt',                                                                                                                                                                              
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',                                                                                                                        
        'outtmpl': f'{video_dir}/game_1/game_1.mp4',                                                                                                                                                         
    }                                                                                                                                                                                           

    # Get video to process.                                                                                                                                                        
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(['https://www.youtube.com/watch?v=LPDnemFoqVk'])  

# End of import_video()

def normalize_sequences(sequences):                                                                                                                                                         
    # Fit scaler on all frames across all sequences
    all_frames = np.concatenate([s['sequence'] for s in sequences], axis=0)                                                                                                                 
    scaler = StandardScaler()                                                                                                                                                               
    scaler.fit(all_frames)
                                                                                                                                                                                            
    for s in sequences:                                                                                                                                                                     
        s['sequence'] = scaler.transform(s['sequence'])
                                                                                                                                                                                            
    return sequences, scaler

# End of normalize_sequences().

def plot_elbow(embeddings, k_min=2, k_max=15):                                                                                                                                              
    inertias = []                                                                                                                                                                           
    silhouettes = []                                                                                                                                                                        
    ks = range(k_min, k_max + 1)

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = km.fit_predict(embeddings)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(embeddings, labels))
                                                                                                                                                                                            
    optimal_k = ks[np.argmax(silhouettes)]
    print(f"Optimal k: {optimal_k} (silhouette={max(silhouettes):.4f})")                                                                                                                    
                                                                                                                                                                                            
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))                                                                                                                                   
                                                                                                                                                                                            
    ax1.plot(ks, inertias, marker='o')                                                                                                                                                      
    ax1.set_xlabel('k')
    ax1.set_ylabel('inertia')
    ax1.set_title('Elbow curve')
                                                                                                                                                                                            
    ax2.plot(ks, silhouettes, marker='o', color='orange')
    ax2.axvline(optimal_k, linestyle='--', color='red', label=f'optimal k={optimal_k}')                                                                                                     
    ax2.set_xlabel('k')                                                                                                                                                                     
    ax2.set_ylabel('silhouette score')
    ax2.set_title('Silhouette score (higher = better)')                                                                                                                                     
    ax2.legend()                                                                                                                                                                            

    plt.tight_layout()                                                                                                                                                                      
    plt.savefig('elbow_curve.png', dpi=150)
    plt.show()

    return optimal_k

# End of plot_elbow().

def semantic_segmentation():
    df = pd.read_csv('my_basket_tube/csv/df_balls.csv', index_col=0)                                                                                                                        
    df = df.sort_values('confidence', ascending=False).drop_duplicates('frame_idx').sort_values('frame_idx').reset_index(drop=True)                                                         
                                                                                                                                                                                            
    df_vectors = compute_motion_vectors(df,  frames_dir=f"{video_dir}/game_1/ball_frames")
    plot_log_magnitude_histogram(df_vectors)                                                                                                                                                 
    df_vectors = magnitude_bucket_analysis(df_vectors)                                                                                                                                      
    df_vectors.to_csv('my_basket_tube/csv/df_vectors_bucketed.csv')                                                                                                                         
                                                                                                                                                                                            
    plot_motion_phase(df_vectors)

# End semantic_segmentation().

def split_video(width=1918, height=1078, fps=30): 
    # Function to split video into frames.  Resolution set for current pipeline architecture.
    '''
        Image dims must be divisible by 14 for DINOv2 patch size operations.
    '''

    # Split video into 1 second frames.                                                   
    frames_dir = Path(f"{video_dir}/game_1/frames")                                                                                                                                                                 
    frames_dir.mkdir(parents=True, exist_ok=True)                                                                                                                                               

    # Note we are no longer sampling assuming 5 frames per second: fps=5. Here we have sampled at broadcast 30fps, the original encoding of the video.                                                                                                                                                                         
    subprocess.run([                                                                                                                                                                            
        "ffmpeg", "-i", f"{video_dir}/game_1/game_1.mp4",
        "-vf", f"fps={fps},scale={width}:{height}",
        str(frames_dir / "%04d.jpg")
    ], check=True) 

# End of split_video()


def extract_embeddings(encoder, sequences, num_frames=20):                                                                                                                                  
    encoder.eval()                                                                                                                                                                          
    dataset = BallTrajectoryDataset(sequences, num_frames=num_frames)
    loader  = DataLoader(dataset, batch_size=len(dataset), shuffle=False)                                                                                                                   
                                                                                                                                                                                            
    with torch.no_grad():                                                                                                                                                                   
        batch = next(iter(loader))                                                                                                                                                          
        embeddings = encoder(batch).numpy()
                                                                                                                                                                                            
    print(f"Embeddings shape: {embeddings.shape}")
    return embeddings

# End of extract_embeddings().

def cluster_and_plot(embeddings, sequences, k=6):
                                                                                                                                                                                            
    # UMAP projection to 2D
    reducer = umap.UMAP(n_components=2, random_state=42)                                                                                                                                    
    Z2 = reducer.fit_transform(embeddings)                                                                                                                                                  

    # KMeans clustering in native 32D space                                                                                                                                                 
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = km.fit_predict(embeddings)                                                                                                                                                     
                
    # Plot                                                                                                                                                                                  
    plt.figure(figsize=(9, 7))
    scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=labels, cmap='tab10', s=80)
    plt.colorbar(scatter, label='cluster')                                                                                                                                                  
                                                                                                                                                                                            
    for i, s in enumerate(sequences):                                                                                                                                                       
        plt.annotate(f"{s['length']}f", (Z2[i, 0], Z2[i, 1]),                                                                                                                               
                    fontsize=6, alpha=0.6)                                                                                                                                                 

    plt.title(f'Ball trajectory clusters (k={k})')                                                                                                                                          
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')                                                                                                                                                                    
    plt.tight_layout()
    plt.savefig('trajectory_clusters.png', dpi=150)
    plt.show()                                                                                                                                                                              

    for c in range(k):                                                                                                                                                                      
        members = [sequences[i] for i in range(len(sequences)) if labels[i] == c]
        lengths = [m['length'] for m in members]                                                                                                                                            
        print(f"Cluster {c}: {len(members)} events | length {min(lengths)}–{max(lengths)} frames | median {int(np.median(lengths))}")
                                                                                                                                                                                            
    return labels, Z2
                                                                                                                                                                                            
    labels, Z2 = cluster_and_plot(embeddings, sequences, k=6)

# End cluster_and_plot().
   
        
def plot_motion_phase(df_vectors, stride=90):                                                                                                                                               
                                                                                                                                                                                                                                                                                                                                          
    bucket_colors = {0: 'gray', 1: 'steelblue', 2: 'crimson'}                                                                                                                               
                                                                                                                                                                                            
    dx = df_vectors['dx'].values                                                                                                                                                            
    dy = df_vectors['dy'].values                                                                                                                                                            
    buckets = df_vectors['magnitude_bucket'].values                                                                                                                                         
                                                                                                                                                                                            
    idx = np.arange(0, len(dx), stride)
    dx_s = dx[idx]
    dy_s = dy[idx]
    buckets_s = buckets[idx]                                                                                                                                                                

    mag = np.sqrt(dx_s**2 + dy_s**2)                                                                                                                                                        
    u = mag * dx_s
    v = mag * dy_s
                                                                                                                                                                                            
    x_pos = np.concatenate([[0], np.cumsum(u[:-1])])
    y_pos = np.concatenate([[0], np.cumsum(v[:-1])])                                                                                                                                        
                
    colors = [bucket_colors[b] for b in buckets_s]                                                                                                                                          

    plt.figure(figsize=(10, 10))                                                                                                                                                            
    for b, label in [(0, 'log-small'), (1, 'log-medium'), (2, 'log-short')]:
        mask = buckets_s == b                                                                                                                                                               
        plt.quiver(x_pos[mask], y_pos[mask], u[mask], v[mask],
                    angles='xy', scale_units='xy', scale=1,                                                                                                                                  
                    color=bucket_colors[b], alpha=0.7, width=0.002, label=label)                                                                                                             

    plt.xlabel('cumulative dx')                                                                                                                                                             
    plt.ylabel('cumulative dy')
    plt.title('Ball motion phase portrait (head-to-tail)')                                                                                                                                  
    plt.legend()
    plt.tight_layout()                                                                                                                                                                      
    plt.savefig('my_basket_tube/csv/motion_phase_w_log_bucketing.png', dpi=150)
    plt.show()                                                                                                                                                                              
    print("Saved: motion_phase_w_log_bucketing.png")
   
# End of plot_motion_phase().

def create_labeled_tracing_videos(df_labeled, frames_dir, output_dir, fps=30, trail_window=120):
    """
    df_labeled: DataFrame with columns ['frame_idx', 'label']                                                                                                                               
    frames_dir: Path to directory containing frame_*.jpg files                                                                                                                              
    output_dir: Path to write one video per unique label: 'magnitude_bucket'                                                                                                                                  
    """                                                                                                                                                                                     
    frames_dir = Path(frames_dir)
    output_dir = Path(output_dir)                                                                                                                                                           
    output_dir.mkdir(parents=True, exist_ok=True)
                                                                                                                                                                                            
    # Build centroid map from df_labeled (expects cx, cy columns)
    centroid_map = dict(zip(df_labeled['frame_idx'], zip(df_labeled['cx'].astype(int), df_labeled['cy'].astype(int))))                                                                      
                                                                                                                                                                                            
    # Preload frame dimensions                                                                                                                                                              
    sample = sorted(frames_dir.glob("frame_*.jpg"))[0]                                                                                                                                      
    h, w = cv2.imread(str(sample)).shape[:2]                                                                                                                                                
                                                                                                                                                                                            
    for bucket in sorted(df_labeled['magnitude_bucket'].unique()):
        df_label = df_labeled[df_labeled['magnitude_bucket'] == bucket].sort_values('frame_idx')                                                                                                        
        frame_indices = set(df_label['frame_idx'].values)                                                                                                                                   

        frame_files = sorted(                                                                                                                                                               
            [f for f in frames_dir.glob("frame_*.jpg")
            if (int(f.stem.split('_')[1]) - 1) in frame_indices],                                                                                                                          
            key=lambda f: int(f.stem.split('_')[1])                                                                                                                                         
        )
                                                                                                                                                                                            
        if not frame_files:
            print(f"Bucket {bucket}: no frames found, skipping.")
            continue                                                                                                                                                                        

        out_path = output_dir / f"bucket_{bucket}.mp4"                                                                                                                                        
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                                                                                                                                                                            
        trail = []
                                                                                                                                                                                            
        for frame_file in tqdm(frame_files, desc=f"Bucket {bucket}"):
            frame_idx = int(frame_file.stem.split('_')[1]) - 1
                                                                                                                                                                                            
            if frame_idx in centroid_map:
                trail.append((frame_idx, centroid_map[frame_idx]))                                                                                                                          
                
            trail = [(fi, pt) for fi, pt in trail if frame_idx - fi <= trail_window]                                                                                                        

            img = cv2.imread(str(frame_file))                                                                                                                                               
            if img is None:
                continue

            if len(trail) >= 2:
                pts = np.array([pt for _, pt in trail], dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [pts], isClosed=False, color=(0, 255, 0), thickness=2)                                                                                                   

            if trail:                                                                                                                                                                       
                cv2.circle(img, trail[-1][1], 6, (0, 0, 255), -1)
                                                                                                                                                                                            
            writer.write(img)
                                                                                                                                                                                            
        writer.release()
        print(f"Bucket {bucket}: saved {out_path} ({len(frame_files)} frames)")

# End of create_tracing_video().

def magnitude_bucket_analysis(df_vectors):                                                                                                                                                  
    df_vectors = df_vectors.copy()
    df_vectors['magnitude'] = np.sqrt(df_vectors['dx']**2 + df_vectors['dy']**2)                                                                                                            
    df_vectors['log_magnitude'] = np.log10(df_vectors['magnitude'].clip(1e-9))                                                                                                              

    # Manual log-scale split points based on observed histogram populations:                                                                                                                
    # Bucket 0: near-zero / stationary  (log10 < -2,  magnitude < 0.01)
    # Bucket 1: normal motion           (-2 <= log10 <= 1, magnitude 0.01–10)                                                                                                               
    # Bucket 2: high-motion transitions (log10 > 1,   magnitude > 10)                                                                                                                       
    split_points = [-2, 1]  # log10 scale                                                                                                                                                   
                                                                                                                                                                                            
    df_vectors['magnitude_bucket'] = np.digitize(
        df_vectors['log_magnitude'], split_points                                                                                                                                           
    )           

    bin_edges_log = [-9] + split_points + [np.inf]                                                                                                                                          
    for b in range(3):
        mask = df_vectors['magnitude_bucket'] == b                                                                                                                                          
        count = mask.sum()
        lo = 10 ** bin_edges_log[b]                                                                                                                                                         
        hi = 10 ** bin_edges_log[b + 1] if bin_edges_log[b + 1] != np.inf else float('inf')
        print(f"Bucket {b}: {count} frames | magnitude [{lo:.4f}, {hi:.4f})")                                                                                                               

    return df_vectors

# End of magnitude_bucket_analysis().

def plot_log_magnitude_histogram(df_vectors):                                                                                                                                               
    magnitudes = np.sqrt(df_vectors['dx']**2 + df_vectors['dy']**2)
    log_mags = np.log10(magnitudes.clip(1e-9))                                                                                                                                              
                                                                                                                                                                                            
    Path('my_basket_tube/plots').mkdir(parents=True, exist_ok=True)
                                                                                                                                                                                            
    plt.figure(figsize=(10, 4))
    plt.hist(log_mags, bins=100, color='steelblue', edgecolor='none')
    plt.xlabel('log10(magnitude)')                                                                                                                                                          
    plt.ylabel('frame count')
    plt.title('Distribution of Normalized Motion Magnitude')                                                                                                                                
    plt.tight_layout()
    plt.savefig('my_basket_tube/plots/log_magnitude_hist.png')                                                                                                                              
    plt.close()
    print("Saved: log_magnitude_hist.png") 



### TEsting

def extract_sequences(df, window_size, stride=1):                                                                                                                                           
    vectors = df[['norm_dx', 'norm_dy']].values
    sequences = []                                                                                                                                                                          
    indices = []
    for i in range(0, len(vectors) - window_size + 1, stride):
        sequences.append(vectors[i:i + window_size])                                                                                                                                        
        indices.append(i)
    return np.array(sequences), indices                                                                                                                                                     
                                                                                                                                                                                            
def dtw_umap_embed(sequences, n_neighbors=15, min_dist=0.1):
    print(f"Computing {len(sequences)}x{len(sequences)} DTW distance matrix...")                                                                                                            
    distance_matrix = dtw_ndim.distance_matrix_fast(sequences)                                                                                                                              
    print("Fitting UMAP...")                                                                                                                                                                
    reducer = umap.UMAP(metric='precomputed', n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)                                                                                  
    embedding = reducer.fit_transform(distance_matrix)                                                                                                                                      
    return embedding
                                                                                                                                                                                            
def run_dtw_umap(df_vectors, window_sizes=[60, 90, 120, 150], subsample=2000, n_neighbors=30):                                                                                              
    df = compute_normed_vectors(df_vectors)                                                                                                                                                 
                                                                                                                                                                                            
    fig, axes = plt.subplots(1, len(window_sizes), figsize=(6 * len(window_sizes), 6))                                                                                                      
                
    bucket_colors = {0: 'gray', 1: 'steelblue', 2: 'crimson'}                                                                                                                               
    bucket_labels = {0: 'stationary', 1: 'normal', 2: 'high-motion'}
                                                                                                                                                                                            
    for ax, window_size in zip(axes, window_sizes):                                                                                                                                         
        print(f"\nWindow size: {window_size} frames ({window_size/30:.1f}s)")
        sequences, indices = extract_sequences(df, window_size)                                                                                                                             
                
        # Get bucket label for each sequence (use center frame)                                                                                                                             
        center_offsets = [i + window_size // 2 for i in indices]
        center_offsets = [min(o, len(df) - 1) for o in center_offsets]                                                                                                                      
        seq_buckets = df['magnitude_bucket'].iloc[center_offsets].values                                                                                                                    
                                                                                                                                                                                            
        if len(sequences) > subsample:                                                                                                                                                      
            idx = np.random.choice(len(sequences), subsample, replace=False)                                                                                                                
            sequences = sequences[idx]
            seq_buckets = seq_buckets[idx]                                                                                                                                                  
            print(f"Subsampled to {subsample} sequences")
                                                                                                                                                                                            
        embedding = dtw_umap_embed(sequences, n_neighbors=n_neighbors)                                                                                                                      

        for b, label in bucket_labels.items():                                                                                                                                              
            mask = seq_buckets == b
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                    s=4, alpha=0.6, c=bucket_colors[b], label=label)                                                                                                                      

        ax.set_title(f'window={window_size} ({window_size/30:.1f}s)')                                                                                                                       
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')                                                                                                                                                             
        ax.legend(markerscale=3, fontsize=8)
                                                                                                                                                                                            
    plt.suptitle('DTW + UMAP — motion sequence embedding')
    plt.tight_layout()                                                                                                                                                                      
    plt.savefig('my_basket_tube/csv/dtw_umap.png', dpi=150)
    plt.show()
    print("Saved: dtw_umap.png")


def compute_normed_vectors(df_vectors):                                                                                                                                                     
    df = df_vectors.copy()
    df['norm_dx'] = df['dx'] / np.sqrt(df['area'])                                                                                                                                          
    df['norm_dy'] = df['dy'] / np.sqrt(df['area'])                                                                                                                                          
    return df

### Run Code.

step = "thinkies"   

match step:

    case 'detect_players':

        detect_players()

    case "create_labeled_tracing_videos":

        # Read in vector dataframe.
        df = pd.read_csv('my_basket_tube/csv/df_vectors_bucketed.csv')

        # Set frames directory.
        frames_dir = "my_basket_tube/videos/game_1/ball_frames"
        output_dir = "my_basket_tube/videos/game_1/log_mag"

        create_labeled_tracing_videos(df, frames_dir=frames_dir, output_dir=output_dir)

    case "motion_phase":                                                                                                                                                                        
      df = pd.read_csv('my_basket_tube/csv/df_balls.csv', index_col=0)                                                                                                                        
      df = df.sort_values('confidence', ascending=False).drop_duplicates('frame_idx').sort_values('frame_idx').reset_index(drop=True)                                                         
      df_vectors = compute_motion_vectors(df)                                                                                                                                                 
      df_vectors['magnitude'] = np.sqrt(df_vectors['dx']**2 + df_vectors['dy']**2)
      print(df_vectors[['frame_idx', 'dx', 'dy', 'magnitude']].to_string())                                                                                                                   
      df_vectors.to_csv('my_basket_tube/csv/df_vectors.csv')                                                                                                                                  
      plot_motion_phase(df_vectors)

    case "tracing": 

        create_tracing_video()


    case "import_video": 

        import_video()
        #parse_subtitles()


    case "split_video":

        split_video()

    case "detect_ball":

        detect_ball()   
    
    case "frequency_analysis":
                                                                                                                                                                                                
        df = pd.read_csv('my_basket_tube/csv/df_balls.csv', index_col=0)                                                                                                                        
        df = df.sort_values('confidence', ascending=False).drop_duplicates('frame_idx').sort_values('frame_idx').reset_index(drop=True)                                                         
        df_vectors = compute_motion_vectors(df)                                                                                                                                                 
                                                                                                                                                                                                
        analyze_frequency(df_vectors)

    case "semantic_segmentation":                                                                                                                                                               
      
        semantic_segmentation()

    case "dtw_umap":

        df = pd.read_csv("my_basket_tube/csv/df_vectors_bucketed.csv")
        run_dtw_umap(df)   

    case "thinkies":

        def detect_dribble_sequences(df_vectors, window_size=10, threshold=0.7):
            dy = df_vectors['dy'].values                                                                                                                                                            
            dy_sign = np.sign(dy)
                                                                                                                                                                                                    
            # Platonic dribble template: alternating up/down                                                                                                                                        
            template = np.array([-1, 1] * (window_size // 2), dtype=float)
            template /= np.linalg.norm(template)                                                                                                                                                    
                        
            scores = []                                                                                                                                                                             
            for i in range(len(dy_sign) - window_size):
                window = dy_sign[i:i + window_size].astype(float)
                norm = np.linalg.norm(window)
                if norm == 0:
                    scores.append(0.0)
                    continue
                score = np.dot(window / norm, template)
                scores.append(abs(score))  # abs: phase-independent                                                                                                                                 
                                                                                                                                                                                                    
            scores = np.array(scores)                                                                                                                                                               
            df_vectors = df_vectors.iloc[:len(scores)].copy()                                                                                                                                       
            df_vectors['dribble_score'] = scores                                                                                                                                                    
            df_vectors['is_dribble'] = scores >= threshold
                                                                                                                                                                                                    
            print(f"Dribble frames: {df_vectors['is_dribble'].sum()} / {len(df_vectors)}")
            return df_vectors 
        
        def plot_dribble_score_distribution(df_vectors):                                                                                                                                            
            scores = df_vectors['dribble_score'].values                                                                                                                                             
                                                                                                                                                                                                    
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))                                                                                                                                         

            axes[0].hist(scores, bins=100, color='steelblue', alpha=0.7)                                                                                                                            
            axes[0].set_xlabel('dribble score')
            axes[0].set_ylabel('count')                                                                                                                                                             
            axes[0].set_title('Dribble score distribution')

            axes[1].plot(scores, linewidth=0.5, color='steelblue', alpha=0.7)                                                                                                                       
            axes[1].set_xlabel('frame')
            axes[1].set_ylabel('dribble score')                                                                                                                                                     
            axes[1].set_title('Dribble score over time')

            plt.tight_layout()                                                                                                                                                                      
            plt.savefig('my_basket_tube/csv/dribble_scores.png', dpi=150)
            plt.show()                                                                                                                                                                              
            print("Saved: dribble_scores.png")

        def plot_phase_unrolled(df_vectors, stride=1):
            df = df_vectors.iloc[::stride].copy()
            magnitude = np.sqrt(df['dx']**2 + df['dy']**2).values
            dy = df['dy'].values                                                                                                                                                                    

            x_pos = np.concatenate([[0], np.cumsum(magnitude[:-1])])                                                                                                                                
            y_pos = np.concatenate([[0], np.cumsum(dy[:-1])])
                                                                                                                                                                                                    
            fig = go.Figure()
            fig.add_trace(go.Scatter(                                                                                                                                                               
                x=x_pos, y=y_pos,
                mode='lines',
                line=dict(color='steelblue', width=0.8),                                                                                                                                            
                text=df['frame_idx'].values,
                hovertemplate='frame: %{text}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>'                                                                                                         
            ))                                                                                                                                                                                      

            fig.update_layout(                                                                                                                                                                      
                title='Phase map — unrolled (no backtracking)',
                xaxis_title='unrolled magnitude',                                                                                                                                                   
                yaxis_title='cumulative dy',
                width=1600, height=600                                                                                                                                                              
            )           

            fig.write_html('my_basket_tube/csv/phase_unrolled.html')                                                                                                                                
            print("Saved: phase_unrolled.html")

        def detect_bounded_regions(df_vectors, window=30, std_threshold=15, min_region_length=20):
            df = df_vectors.copy()                                                                                                                                                                  
            dy = df['dy'].values
            df['cumulative_dy'] = np.concatenate([[0], np.cumsum(dy[:-1])])                                                                                                                         
                                                                                                                                                                                                    
            cum_dy = pd.Series(df['cumulative_dy'].values)
                                                                                                                                                                                                    
            # Detrend locally: subtract rolling mean, measure std of residual                                                                                                                       
            rolling_mean = cum_dy.rolling(window, center=True, min_periods=1).mean()
            residual = cum_dy - rolling_mean                                                                                                                                                        
            rolling_std = residual.rolling(window, center=True, min_periods=1).std()
                                                                                                                                                                                                    
            df['rolling_std'] = rolling_std
            df['in_region'] = rolling_std <= std_threshold                                                                                                                                          
                        
            df['region_id'] = (df['in_region'] != df['in_region'].shift(1)).cumsum()                                                                                                                
            df.loc[~df['in_region'], 'region_id'] = -1
                                                                                                                                                                                                    
            regions = df[df['in_region']].groupby('region_id').filter(lambda x: len(x) >= min_region_length)
            n_regions = regions['region_id'].nunique()
            print(f"Found {n_regions} bounded regions (std_threshold={std_threshold})")                                                                                                             

            return df, n_regions 


        def plot_phase_unrolled_regions(df_vectors, stride=1):                                                                                                                                      
            df = df_vectors.iloc[::stride].copy()                                                                                                                                                   
            magnitude = np.sqrt(df['dx']**2 + df['dy']**2).values                                                                                                                                   
            dy = df['dy'].values                                                                                                                                                                    
                        
            x_pos = np.concatenate([[0], np.cumsum(magnitude[:-1])])                                                                                                                                
            y_pos = np.concatenate([[0], np.cumsum(dy[:-1])])
                                                                                                                                                                                                    
            fig = go.Figure()

            # Base trace                                                                                                                                                                            
            fig.add_trace(go.Scatter(
                x=x_pos, y=y_pos,                                                                                                                                                                   
                mode='lines',
                line=dict(color='steelblue', width=0.8),
                text=df['frame_idx'].values,                                                                                                                                                        
                hovertemplate='frame: %{text}<br>x: %{x:.1f}<br>y: %{y:.1f}<extra></extra>',
                name='trajectory'                                                                                                                                                                   
            ))          
                                                                                                                                                                                                    
            # Overlay detected regions                                                                                                                                                              
            colors = ['rgba(255,100,100,0.3)', 'rgba(100,255,100,0.3)',
                    'rgba(255,165,0,0.3)', 'rgba(180,100,255,0.3)']                                                                                                                               
                        
            for i, (region_id, group) in enumerate(df[df['region_id'] > 0].groupby('region_id')):                                                                                                   
                idx = group.index
                color = colors[i % len(colors)]                                                                                                                                                     
                fig.add_trace(go.Scatter(                                                                                                                                                           
                    x=x_pos[idx], y=y_pos[idx],
                    mode='lines',                                                                                                                                                                   
                    line=dict(width=3, color=color.replace('0.3', '1.0')),
                    fill='tozeroy',                                                                                                                                                                 
                    fillcolor=color,
                    hovertemplate=f'region {region_id}<br>frame: %{{text}}<extra></extra>',                                                                                                         
                    text=group['frame_idx'].values,                                                                                                                                                 
                    name=f'region {region_id}'
                ))                                                                                                                                                                                  
                        
            fig.update_layout(                                                                                                                                                                      
                title='Phase map — unrolled with detected regions',
                xaxis_title='unrolled magnitude',                                                                                                                                                   
                yaxis_title='cumulative dy',                                                                                                                                                        
                width=1600, height=600
            )                                                                                                                                                                                       
                        
            fig.write_html('my_basket_tube/csv/phase_unrolled_regions.html')                                                                                                                        
            print("Saved: phase_unrolled_regions.html")

        # df = pd.read_csv("my_basket_tube/csv/df_vectors.csv")
        # df_vectors, n_regions = detect_bounded_regions(df)                                                                                                                                  
        # plot_phase_unrolled_regions(df_vectors)

        import whisper                                                                                                                                                                              
                                                                                                                                                                                              
        def transcribe_video():                                                                                                                                                                     
            model = whisper.load_model("medium")
            result = model.transcribe(f"{video_dir}/game_1/game_1.mp4")                                                                                                                             
                                                                                                                                                                                                    
            rows = []
            for segment in result['segments']:                                                                                                                                                      
                rows.append({                                                                                                                                                                       
                    'start': segment['start'],
                    'end': segment['end'],                                                                                                                                                          
                    'start_frame': int(segment['start'] * 30),
                    'end_frame': int(segment['end'] * 30),                                                                                                                                          
                    'text': segment['text'].strip()
                })                                                                                                                                                                                  
                        
            df = pd.DataFrame(rows)
            df.to_csv('my_basket_tube/csv/subtitles.csv')
            print(f"Transcribed {len(df)} segments")                                                                                                                                                
            return df   
        
        transcribe_video()





        
                                                                                                                                                                                              




