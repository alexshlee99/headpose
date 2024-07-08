import os
import pandas as pd 


# Input data: create csv (vid_name, annotation labels, length)
cropped_aligned_data_pth = "/home/alex/바탕화면/Data/Aff-Wild2/cropped_aligned"

df = pd.DataFrame(columns=['vid_name', 'Expression', 'Valence', 'Arousal', 'AU'])

# for vid in sorted(os.listdir(cropped_aligned_data_pth)): 
#     if not vid.startswith('.'):     
#         vid_pth = os.path.join(cropped_aligned_data_pth, vid)       
#         for img in sorted(os.listdir(vid_pth)): 
#             if not img.startswith('.'): 
#                 row = {'vid_name': vid, 
        

