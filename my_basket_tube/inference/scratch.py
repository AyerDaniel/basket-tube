import pandas as pd                                                           
df = pd.read_csv('my_basket_tube/csv/play_by_play_frames.csv')                
print(df.shape)                                               
print(df['frame_id'].isna().sum(), 'missing frame_ids')                       
print(df['frame_id'].notna().sum(), 'matched')         
print(df.head(10).to_string()) 


# Test
def gameclock_to_frame(quarter, game_clock):                                                                                                                                                
      """Convert quarter + game clock string (MM:SS) to approximate frame number."""                                                                                                        
      quarter_anchors = {                                                                                                                                                                     
          1: 9481,                                                                                                                                                                          
          2: 49771,                                                                                                                                                                           
          3: 76531,                                                                                                                                                                         
          4: 114931,                                                                                                                                                                          
      }                                                                                                                                                                                     
      minutes, seconds = map(float, game_clock.split(':'))
      clock_seconds = minutes * 60 + seconds
      seconds_elapsed = 720 - clock_seconds  # clock counts down from 12:00                                                                                                                   
      frame = quarter_anchors[quarter] + int(seconds_elapsed * 30)                                                                                                                            
      return frame      

# End gameclock_to_frame().

# frame = gameclock_to_frame(3,"7:30.0")
# import cv2                                                                                                                                                                                  
# img = cv2.imread(f"my_basket_tube/videos/game_1/frames/{frame}.jpg")
# cv2.imshow('frame', img)
# cv2.waitKey(0)


## End Testing


def semantic_segmentation():
    # This function takes in the still frames containing the ball, compute smotion vectors, and then uses those as markers for semantic event delineation.

    # Read in detection csv.
    df = pd.read_csv('my_basket_tube/csv/df_balls.csv', index_col=0)     

    # Sort detections chronologically.                                                                                                                                          
    df = df.sort_values('confidence', ascending=False).drop_duplicates('frame_idx').sort_values('frame_idx').reset_index(drop=True)

    # Caclulcate trajectories.                                                                                                                                                                  
    df['cx'] = (df['x1'] + df['x2']) / 2                                                                                                                                                    
    df['cy'] = (df['y1'] + df['y2']) / 2                                                                                                                                                    
    df['area'] = (df['x2'] - df['x1']) * (df['y2'] - df['y1'])                                                                                                                              
                                                                                                                                                                                            
    # Z is the "embedding" — raw trajectory features                                                                                                                                       
    Z = df[['cx', 'cy', 'area', 'confidence']].values                                                                                                                                       
                                                                                                                                                                                            
    # Frame-to-frame L2 distance   <-- We're using this to find large transitons.                                                                                                                                                         
    d = np.linalg.norm(Z[1:] - Z[:-1], axis=1)
    d = np.concatenate([[0.0], d])                                                                                                                                                          
                
    
    # # Original threshold was 98th percentile.  However, this did not identify semantic events well.
    # # Spikes above 98th percentile = boundaries                                                                                                                                             
    # thr = float(np.quantile(d, 0.98))

    # Try different threshold.
    thr = float(np.quantile(d, 0.75))

    MIN_GAP = 30                                                                                                                                           
    boundaries = []
    last = -10**9                                                                                                                                                                           
    for i, val in enumerate(d):
        if val >= thr and i - last >= MIN_GAP:                                                                                                                                              
            boundaries.append(i)
            last = i            

    # Get frame_ids of boundaries.
    frame_ids = [df['frame_idx'].iloc[x] for x in boundaries]

    # Extract sequence for each event                                                                                                                                                       
    sequences = []
    for i in range(len(frame_ids) - 1):
        start_frame = frame_ids[i]                                                                                                                                                          
        end_frame   = frame_ids[i + 1]
                                                                                                                                                                                            
        chunk = df[(df['frame_idx'] >= start_frame) & (df['frame_idx'] <= end_frame)].copy()                                                                                                

        if len(chunk) < 2:                                                                                                                                                                  
            continue

        # Relative motion — position invariant                                                                                                                                              
        chunk['dx'] = chunk['cx'].diff()
        chunk['dy'] = chunk['cy'].diff()                                                                                                                                                    
        chunk['d_area'] = chunk['area'].diff()
        chunk = chunk.dropna()                                                                                                                                                              
                
        seq = chunk[['dx', 'dy', 'd_area', 'confidence']].values                                                                                                                            

        sequences.append({                                                                                                                                                                  
            'event_id':    i,
            'start_frame': start_frame,                                                                                                                                                     
            'end_frame':   end_frame,
            'length':      len(seq),                                                                                                                                                        
            'sequence':    seq,                                                                                                                                                             
        })
                                                                                                                                                                                            
    print(f"Extracted {len(sequences)} event sequences")
    print(f"Length range: {min(s['length'] for s in sequences)} to {max(s['length'] for s in sequences)} frames")
    print(f"Median length: {np.median([s['length'] for s in sequences]):.0f} frames")                                                                                                       
                                                                                                                                                                                            
    return sequences

    # Testing    
    show_plots = False

    if show_plots:
        # Plot Event Boundaries.                                                                                                                                                                                
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
                                                                                                                                                                                                
        axes[0].plot(df['frame_idx'], df['cx'], alpha=0.7)
        axes[0].set_ylabel('cx (pixels)')                                                                                                                                                       
        axes[0].set_title('Ball centroid X')                                                                                                                                                    

        axes[1].plot(df['frame_idx'], df['cy'], alpha=0.7, color='orange')                                                                                                                      
        axes[1].set_ylabel('cy (pixels)')
        axes[1].set_title('Ball centroid Y')                                                                                                                                                    
                                                                                                                                                                                                
        axes[2].plot(df['frame_idx'].values, d, color='red', alpha=0.8)                                                                                                                         
        axes[2].axhline(thr, linestyle='--', color='black', label=f'threshold={thr:.1f}')                                                                                                       
        axes[2].set_ylabel('change-point score')                                                                                                                                                
        axes[2].set_xlabel('frame_idx')
        axes[2].set_title('Frame-to-frame L2 distance (spikes = event boundaries)')                                                                                                             
        axes[2].legend()                                                                                                                                                                        

        for b in boundaries:                                                                                                                                                                    
            frame = df['frame_idx'].iloc[b]
            for ax in axes:                                                                                                                                                                     
                ax.axvline(frame, linestyle='--', alpha=0.4, color='green')                                                                                                                     

        plt.tight_layout()                                                                                                                                                                      
        plt.savefig('changepoint_plot.png', dpi=150)
        plt.show()                                                                                                                                                                              

        print(f"Detected {len(boundaries)} boundaries at frames: {[df['frame_idx'].iloc[b] for b in boundaries]}") 

# End of semantic_segmentation().

## TEsting dataset creation.
import pandas as pd 

df = pd.read_csv("my_basket_tube/csv/detections_all.csv")

print(df.info())
print(df.head())

print(df['class'].value_counts())                                                                                                                                                           
numbers = df[df['class']=='number']                                                                                                                                                         
players = df[df['class']=='player']                                                                                                                                                         
print(f'Detail hit rate: {df.detail.notna().sum()} / {len(numbers)}')
print(f'Team hit rate: {df.team.notna().sum()} / {len(players)}')   

