import csv                                                                                                                                                                                  
import os       
from collections import defaultdict

# Config
CONTEXT_BEFORE = 150  # 5 seconds
CONTEXT_AFTER  = 60   # 2 seconds                                                                                                                                                           
FRAMES_DIR     = 'videos/game_1/frames'
                                                                                                                                                                                            
# Load roster — name lookup: "S. Curry" -> (team, number)                                                                                                                                   
roster = {}                                                                                                                                                                                 
with open('my_basket_tube/csv/roster.csv') as f:                                                                                                                                            
    for row in csv.DictReader(f):                                                                                                                                                           
        key = f"{row['initial']} {row['last_name']}"
        roster[key] = {'team': row['team'], 'number': int(row['number'])}                                                                                                                   
                
# Load tracks — index by (team, number, frame_id) -> box                                                                                                                                    
print('Loading tracks...')
tracks = defaultdict(dict)  # tracks[(team, number)][frame_id] = (x1,y1,x2,y2)                                                                                                              
with open('my_basket_tube/csv/tracks.csv') as f:                                                                                                                                            
    for row in csv.DictReader(f):                                                                                                                                                           
        if not row['player_number']:                                                                                                                                                        
            continue                                                                                                                                                                        
        key = (row['team'], int(float(row['player_number'])))
        frame = int(row['frame_id'])                                                                                                                                                        
        tracks[key][frame] = (
            int(row['x1']), int(row['y1']),                                                                                                                                                 
            int(row['x2']), int(row['y2'])
        )                                                                                                                                                                                   
                                                                                                                                                                                            
# Load ball positions — index by frame_id
print('Loading ball detections...')                                                                                                                                                         
balls = {}      
with open('my_basket_tube/csv/df_balls.csv') as f:                                                                                                                                          
    for row in csv.DictReader(f):
        frame = int(row['frame_idx'])                                                                                                                                                       
        balls[frame] = (                                                                                                                                                                    
            (int(row['x1']) + int(row['x2'])) // 2,
            (int(row['y1']) + int(row['y2'])) // 2,                                                                                                                                         
            float(row['confidence'])
        )                                                                                                                                                                                   
                
# Load play-by-play (stub: use estimated frame_id if real one not available)                                                                                                                
print('Loading play-by-play...')
pbp_file = 'my_basket_tube/csv/play_by_play_frames.csv'                                                                                                                                     
if not os.path.exists(pbp_file):                                                                                                                                                            
    pbp_file = 'my_basket_tube/csv/play_by_play_clean.csv'  # stub fallback
                                                                                                                                                                                            
SHOT_ACTIONS = {'made_shot', 'missed_shot'}                                                                                                                                                 
                                                                                                                                                                                            
sequences = []                                                                                                                                                                              
                
with open(pbp_file) as f:
    for row in csv.DictReader(f):
        action = row['action_type']                                                                                                                                                         
        player_name = row['player']
        team = row['team']                                                                                                                                                                  
                                                                                                                                                                                            
        if not player_name or player_name not in roster:
            continue                                                                                                                                                                        
                
        # Get frame anchor
        frame_id = int(row['frame_id']) if row.get('frame_id') else None
        if frame_id is None:                                                                                                                                                                
            continue
                                                                                                                                                                                            
        # Resolve player to track key
        r = roster[player_name]                                                                                                                                                             
        track_key = (r['team'], r['number'])
                                                                                                                                                                                            
        # Build sequence
        sequence = []                                                                                                                                                                       
        start = frame_id - CONTEXT_BEFORE
        end   = frame_id + CONTEXT_AFTER                                                                                                                                                    

        for f in range(start, end + 1):                                                                                                                                                     
            # Player box
            box = tracks[track_key].get(f, (0, 0, 0, 0))                                                                                                                                    

            # Ball position                                                                                                                                                                 
            ball = balls.get(f, (0, 0, 0.0))
                                                                                                                                                                                            
            # Check frame exists on disk
            frame_path = os.path.join(FRAMES_DIR, f'frame_{f:04d}.jpg')                                                                                                                     
            if not os.path.exists(frame_path):                                                                                                                                              
                sequence.append((f, box, ball, frame_path, False))
            else:                                                                                                                                                                           
                sequence.append((f, box, ball, frame_path, True))
                                                                                                                                                                                            
        label = 1 if action in SHOT_ACTIONS else 0
                                                                                                                                                                                            
        sequences.append({
            'event':    row,
            'label':    label,                                                                                                                                                              
            'action':   action,
            'sequence': sequence,                                                                                                                                                           
        })                                                                                                                                                                                  

print(f'Built {len(sequences)} sequences')                                                                                                                                                  
print(f'Shot sequences: {sum(1 for s in sequences if s["label"] == 1)}')
print(f'Other sequences: {sum(1 for s in sequences if s["label"] == 0)}')  