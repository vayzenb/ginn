# %%
'''
Creates covariate file for face presence and proportion of face area in each frame of a video
'''

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')
import os
import numpy as np
import pandas as pd
import ginn_params as params
import face_recognition
from PIL import Image
import matplotlib.pyplot as plt
from glob import glob

# %%
exp = 'hbn'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

frame_dir = f'{curr_dir}/stim/fmri_videos/frames/{vid}'

frame_list = glob(f'{frame_dir}/*')
#frame_list = frame_list[0:50]

print(vid)

# %%
frame_covs = pd.DataFrame(columns = ['frame', 'present', 'proportion'])
fn = 1
for frame in range(fn,len(frame_list)+1):

    image = face_recognition.load_image_file(f'{frame_dir}/{vid}_{frame}.jpg')
    face_locations = face_recognition.face_locations(image,model ='cnn')
    if len(face_locations) == 0:
        f_present = 0 #no faces
        prop = 0
    else:
        f_present = 1
        #calculate area of face locations
        face_areas = []
        for face in face_locations:
            face_areas.append((face[2]-face[0])*(face[1]-face[3]))
        
        prop = np.sum(face_areas)/np.prod(image.shape[0:2])
    frame_covs = frame_covs.append({'frame':frame, 'present':f_present, 'proportion':prop}, ignore_index=True)
    #print progress
    print(f'Frame {frame} of {len(frame_list)}')
    

frame_covs.to_csv(f'{curr_dir}/stim/fmri_videos/{vid}_face_covs.csv', index=False)



