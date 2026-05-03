import os                                                                                                                                                                                   
import sys                                                                                                                                                                                  
                                                                                                                                                                                            
sys.path.insert(0, os.path.dirname(__file__))
                                                                                                                                                                                            
import pandas as pd
import vlm_variable_context_window as vlm
                                                                                                                                                                                            
CONTEXT_WINDOWS = [3, 5, 10, 15, 20]
                                                                                                                                                                                            
SUBTITLES_CSV = "my_basket_tube/csv/subtitles.csv"                                                                                                                                          
FRAMES_DIR    = "my_basket_tube/videos/game_1/frames"
VIDEO_PATH    = "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/videos/game_1/game_1.mp4"                                                                               
                                                                                                                                                                                            
df_subtitles = pd.read_csv(SUBTITLES_CSV)                                                                                                                                                   
                                                                                                                                                                                            
for context_seconds in CONTEXT_WINDOWS:                                                                                                                                                     
    print(f"\n{'='*60}")
    print(f"Context window: {context_seconds}s")
    print(f"{'='*60}")                                                                                                                                                                      

    # Override config in vlm module                                                                                                                                                         
    vlm.CONTEXT_SECONDS = context_seconds
    vlm.MODEL_DIR       = vlm.Path(f"my_basket_tube/models/context_{context_seconds}s")                                                                                                     
    vlm.RESULTS_DIR     = vlm.Path(f"my_basket_tube/videos/query_results/context_{context_seconds}s")                                                                                       
                                                                                                                                                                                            
    vlm.train(df_subtitles, FRAMES_DIR)                                                                                                                                                     
    vlm.build_embedding_index(df_subtitles, FRAMES_DIR)                                                                                                                                     
                                                                                                                                                                                            
print("\nSweep complete. All models trained and indexed.")  