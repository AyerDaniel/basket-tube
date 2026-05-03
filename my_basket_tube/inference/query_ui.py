import pandas as pd
import vlm_variable_context_window as vlm
from pathlib import Path

CONTEXT_WINDOWS = [3, 5, 10, 15, 20]
FRAMES_DIR = "my_basket_tube/videos/game_1/frames"
VIDEO_PATH = "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/videos/game_1/game_1.mp4"

def run_query(query):
    for context_seconds in CONTEXT_WINDOWS:
        vlm.CONTEXT_SECONDS = context_seconds
        vlm.MODEL_DIR       = Path(f"my_basket_tube/models/context_{context_seconds}s")
        vlm.RESULTS_DIR     = Path(f"my_basket_tube/videos/query_results/context_{context_seconds}s")
        vlm.query_video(query)

if __name__ == "__main__":
    print("BasketTube Query UI")
    print("Type 'quit' to exit\n")
    while True:
        query = input("Query: ").strip()
        if query.lower() == 'quit':
            break
        if query:
            run_query(query)