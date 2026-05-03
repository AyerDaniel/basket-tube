import os                                                                                                                                                                                   
import re
import cv2                                                                                                                                                                                  
import easyocr  

frames_dir = 'my_basket_tube/videos/game_1/frames'                                                                                                                                                         
crop_box = (886, 1012, 1075, 1051)     # x1, y1, x2, y2
step = 30  # every 1 seconds                                                                                                                                                              
                                                                                                                                                                                            
reader = easyocr.Reader(['en'], gpu=True)                                                                                                                                                   
quarter_hits = {}                                                                                                                                                                           
                                                                                                                                                                                            
frame_files = sorted(                                                                                                                                                                       
      (f for f in os.listdir(frames_dir) if f.endswith('.jpg')),                                                                                                                              
      key=lambda f: int(re.search(r'\d+', f).group())
  )                                                                                                                                                                                           
                                                                                                                  
                                                                                                                                                                                            
for fname in frame_files[::step]:                                                                                                                                                           
    frame_id = int(re.search(r'\d+', fname).group())
    img = cv2.imread(os.path.join(frames_dir, fname))                                                                                                                                       
    x1, y1, x2, y2 = crop_box
    crop = img[y1:y2, x1:x2]                                                                                                                                                                
    results = reader.readtext(crop, detail=0)                                                                                                                                               
    text = ' '.join(results)
    match = re.search(r'(1st|2nd|3rd|4th)', text, re.IGNORECASE)                                                                                                                            
    if match:                                                                                                                                                                               
        quarter = match.group(1).lower()                                                                                                                                                    
        if quarter not in quarter_hits:                                                                                                                                                     
            quarter_hits[quarter] = frame_id
            print(f'{quarter} first seen at frame {frame_id}') 

        if quarter == "4th":
             print('\nSummary:', quarter_hits) 
             break


print('\nSummary:', quarter_hits)  