johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$  source /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/activate
/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py
Saved: log_magnitude_hist.png
Bucket 0: 528 frames | magnitude [0.0000, 0.0100)
Bucket 1: 19424 frames | magnitude [0.0100, 10.0000)
Bucket 2: 1963 frames | magnitude [10.0000, inf)
Saved: motion_phase.png
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py
Traceback (most recent call last):
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py", line 624, in <module>
    df = pd.read_csv("df_vectors_butcketed.csv)")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
             ^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'df_vectors_butcketed.csv)'
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py
Traceback (most recent call last):
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py", line 624, in <module>
    df = pd.read_csv("my_basket_tube/csv/df_vectors_bucketed.csv)")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/parsers/readers.py", line 1880, in _make_engine
    self.handles = get_handle(
                   ^^^^^^^^^^^
  File "/home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/lib/python3.12/site-packages/pandas/io/common.py", line 873, in get_handle
    handle = open(
             ^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'my_basket_tube/csv/df_vectors_bucketed.csv)'
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/main.py
[youtube] Extracting URL: https://www.youtube.com/watch?v=LPDnemFoqVk
[youtube] LPDnemFoqVk: Downloading webpage
WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use another runtime add  --js-runtimes RUNTIME[:PATH]  to your command/config. YouTube extraction without a JS runtime has been deprecated, and some formats may be missing. See  https://github.com/yt-dlp/yt-dlp/wiki/EJS  for details on installing one
[youtube] LPDnemFoqVk: Downloading android vr player API JSON
[info] LPDnemFoqVk: Downloading 1 format(s): 399+140
[info] There are no subtitles for the requested languages
[download] my_basket_tube/videos/game_1/game_1.mp4 has already been downloaded
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/vlm.py
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/vlm.py
Loading CLIP...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Loading weights: 100%|██████████████████████████████████████████████████████| 590/590 [00:00<00:00, 5717.03it/s, Materializing param=visual_projection.weight]
CLIPModel LOAD REPORT from: openai/clip-vit-large-patch14
Key                                  | Status     |  | 
-------------------------------------+------------+--+-
vision_model.embeddings.position_ids | UNEXPECTED |  | 
text_model.embeddings.position_ids   | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading weights: 100%|██████████████████████████████████████████████████████| 590/590 [00:00<00:00, 5258.45it/s, Materializing param=visual_projection.weight]
CLIPModel LOAD REPORT from: openai/clip-vit-large-patch14
Key                                  | Status     |  | 
-------------------------------------+------------+--+-
vision_model.embeddings.position_ids | UNEXPECTED |  | 
text_model.embeddings.position_ids   | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loaded trained projection weights.
Dataset: 1153 subtitle segments

--- Subtitle labels ---
  [0] Baysmore steps back three-pointer way off rebound looney, but he's called for a push
  [1] Who's had that look on opponent's faces after he knocks down threes the same look as lebron james with a clutch shot
  [2] You have to have guys that you trust that when is when it matters most I know exactly what they're going to do
  [3] But what they've done defensively all year to be the best team in the league at that end is special

--- Encoding frames ---
  Visual embedding shape: torch.Size([4, 768])

--- Embedding subtitles via CLIP text encoder ---
  Text embedding shape: torch.Size([4, 768])

--- Cosine similarity (before training) ---
  [0] 0.5693
  [1] 0.6423
  [2] 0.5390
  [3] 0.6113

Sanity check passed.
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ /home/johnsmith/Desktop/njit/workspaces/basket-tube/.venv/bin/python /home/johnsmith/Desktop/njit/workspaces/basket-tube/my_basket_tube/inference/vlm.py
Loading CLIP...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
The image processor of type `CLIPImageProcessor` is now loaded as a fast processor by default, even if the model checkpoint was saved with a slow processor. This is a breaking change and may produce slightly different outputs. To continue using the slow processor, instantiate this class with `use_fast=False`. 
Loading weights: 100%|██████████████████████████████████████████████████████| 590/590 [00:00<00:00, 5739.63it/s, Materializing param=visual_projection.weight]
CLIPModel LOAD REPORT from: openai/clip-vit-large-patch14
Key                                  | Status     |  | 
-------------------------------------+------------+--+-
text_model.embeddings.position_ids   | UNEXPECTED |  | 
vision_model.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Loading weights: 100%|██████████████████████████████████████████████████████| 590/590 [00:00<00:00, 5492.11it/s, Materializing param=visual_projection.weight]
CLIPModel LOAD REPORT from: openai/clip-vit-large-patch14
Key                                  | Status     |  | 
-------------------------------------+------------+--+-
text_model.embeddings.position_ids   | UNEXPECTED |  | 
vision_model.embeddings.position_ids | UNEXPECTED |  | 

Notes:
- UNEXPECTED    :can be ignored when loading from different task/architecture; not ok if you expect identical arch.
Resuming from checkpoint.
Dataset: 1153 subtitle segments
Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 145/145 [01:14<00:00,  1.94it/s]
Epoch 1 loss: 0.4129
Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 145/145 [01:12<00:00,  2.00it/s]
Epoch 2 loss: 0.4131
Epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 145/145 [01:13<00:00,  1.97it/s]
Epoch 3 loss: 0.4102
Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 145/145 [01:13<00:00,  1.97it/s]
Epoch 4 loss: 0.4085
Epoch 5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 145/145 [01:14<00:00,  1.96it/s]
Epoch 5 loss: 0.4078
Epoch 6: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 145/145 [01:14<00:00,  1.95it/s]
Epoch 6 loss: 0.4086
Saved: my_basket_tube/models/projection.pt
(.venv) johnsmith@ravensNest:~/Desktop/njit/workspaces/basket-tube$ 