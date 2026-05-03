import csv                                                                                                                                                                                  
import re                                                                                                                                                                                   
                                                                                                                                                                                            
input_file = 'my_basket_tube/csv/play_by_play.csv'
output_file = 'my_basket_tube/csv/play_by_play_clean.csv'                                                                                                                                           
         
quarter = 1
rows = []
quarter_marker = re.compile(r'^\d(?:st|nd|rd|th) Q$')                                                                                                                                       

def action_list():                                                                                                                                                                                 
                                                                                                                                                                                              
  input_file = '/workspace/play_by_play.csv'                                                                                                                                                  
                                                                                                                                                                                              
  PLAYER_RE = r"[A-Z]\. [A-ZÀ-ÿ][a-zA-ZÀ-ÿ]+(?:[-'][A-Z][a-zA-ZÀ-ÿ]+)*"                                                                                                                       
   
  templates = set()                                                                                                                                                                           
                  
  with open(input_file, 'r', encoding='utf-8') as f:
      for line in f:
          fields = line.rstrip('\n').split('\t')
          for col in [fields[1] if len(fields) > 1 else '',                                                                                                                                   
                      fields[5] if len(fields) > 5 else '']:                                                                                                                                  
              text = col.strip()                                                                                                                                                              
              if text:                                                                                                                                                                        
                  template = re.sub(PLAYER_RE, '{PLAYER}', text)                                                                                                                              
                  templates.add(template)

  for t in sorted(templates):                                                                                                                                                                 
      print(t)

with open(input_file, 'r', encoding='utf-8') as f:                                                                                                                                          
    for line in f:
        line = line.rstrip('\n')                                                                                                                                                            
        fields = line.split('\t')
                                                                                                                                                                                            
        if quarter_marker.match(fields[0].strip()):
            quarter += 1
            continue
                                                                                                                                                                                            
        if fields[0].strip() == 'Time':
            continue                                                                                                                                                                        
                
        while len(fields) < 6:
            fields.append('')

        time  = fields[0].strip()                                                                                                                                                           
        score = fields[3].strip()
                                                                                                                                                                                            
        gs_action = fields[1].strip()
        gs_points = fields[2].strip()
        lal_action = fields[5].strip()
        lal_points = fields[4].strip()                                                                                                                                                      

        if gs_action:                                                                                                                                                                       
            rows.append({'quarter': quarter, 'time': time, 'team': 'golden_state', 'action': gs_action, 'points': gs_points, 'score': score})
                                                                                                                                                                                            
        if lal_action:
            rows.append({'quarter': quarter, 'time': time, 'team': 'la_lakers', 'action': lal_action, 'points': lal_points, 'score': score})                                                
                                                                                                                                                                                            
with open(output_file, 'w', newline='', encoding='utf-8') as f:                                                                                                                             
    fieldnames = ['quarter', 'time', 'team', 'action', 'points', 'score']                                                                                                                   
    writer = csv.DictWriter(f, fieldnames=fieldnames)                                                                                                                                       
    writer.writeheader()                                                                                                                                                                    
    writer.writerows(rows)
                                                                                                                                                                                            
print(f'Written {len(rows)} rows to {output_file}')
