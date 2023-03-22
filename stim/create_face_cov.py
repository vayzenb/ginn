# %%
'''
Creates covariate file for face presence and proportion of face area in each frame of a video
'''

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')
sys.path.insert(1, f'{curr_dir}/modelling')
import os
import numpy as np
import pandas as pd
import ginn_params as params

from PIL import Image
import matplotlib.pyplot as plt
from glob import glob
from scipy.stats import gamma

import pdb

print('libraries loaded')
# %%
exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, fmri_suf, start_trs,end_trs, data_dir, vols, tr, fps, bin_size, ages= params.load_params(exp)

frame_dir = f'{curr_dir}/stim/fmri_videos/frames/{vid}'

frame_list = glob(f'{frame_dir}/*')
#frame_list = frame_list[0:50]
steps = 1
print(vid)

def extract_frames(vid):
    import face_recognition
    # %%
    frame_covs = pd.DataFrame(columns = ['frame', 'present', 'proportion'])
    fn = 1
    for frame in range(fn,len(frame_list)+1,steps):

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
        

        #save every 500 frames
        if frame % 500 == 0:
            frame_covs.to_csv(f'{curr_dir}/stim/fmri_videos/{vid}_face_covs.csv', index=False)


def down_sample(data):
    """Downsample data"""
    downsample_ts = np.empty((0, data.shape[1])) 

    
    #Bin frame data for TS
    for nn in range(0,len(data),bin_size):
        temp = data[nn:(nn+bin_size),:]

        downsample_ts = np.vstack((downsample_ts,np.mean(temp, axis=0)))

    #downsample_ts = downsample_ts[0:(vols-fix_tr),:] #extract only 168 volumes to match fmri data (credits of movie were cut)
    
    return downsample_ts


# %%
'''
Convolve with HRF using
Using double-gamma with 4-sec peak
'''

def hrf(data, tr):
    """ Return values for HRF at given times """
    times = np.arange(0, 30, tr)
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    hrf_at_trs = values / np.max(values) * 0.6 
    
    
    return np.convolve(data, hrf_at_trs)

# %%
def convolve_hrf(data):

    
    conv_ts =np.zeros((data.shape))
    for ii in range(0,data.shape[1]):
        temp = hrf(data[:,ii],tr)

        temp = temp[0:(vols)] # only grab the first 90 volumes
        
        #temp = (temp - np.mean(temp))/np.std(temp)
        conv_ts[:,ii] = temp
        
    return conv_ts


def convolve_face_cov(vid, col):
    #load face cov
    face_cov = pd.read_csv(f'{curr_dir}/stim/fmri_videos/{vid}_face_covs.csv')
    
    #extract desired column
    #convert to numpy
    face_cov = face_cov[col].to_numpy()
    face_cov = face_cov.reshape((len(face_cov),1))
    

    downsample_ts = down_sample(face_cov)

    #add burn volumes to beginning and end of timeseries as needed
    if start_trs > 0:
        downsample_ts = np.vstack((np.zeros((start_trs,downsample_ts.shape[1])), downsample_ts))    

    if end_trs > 0:
        downsample_ts = np.vstack((downsample_ts, np.zeros((end_trs,downsample_ts.shape[1]))))
        
    
    hrf_ts = convolve_hrf(downsample_ts)

    return downsample_ts, hrf_ts
    

downsample_ts, hrf_ts = convolve_face_cov(vid, 'proportion')

#min-max normalize downsampled
downsample_ts = (downsample_ts - np.min(downsample_ts))/(np.max(downsample_ts)-np.min(downsample_ts))

#round downsampled
downsample_ts = np.round(downsample_ts,2)



#save downsampled as txt file
np.savetxt(f'{curr_dir}/fmri/pre_proc/{vid}_face_covs_downsampled.txt', downsample_ts, fmt='%1.3f')
#save convolved as txt file
np.savetxt(f'{curr_dir}/fmri/pre_proc/{vid}_face_covs_convolved.txt', hrf_ts, delimiter=',')


""" 
# grab top responses
binarized = hrf_ts
binarized[binarized>=.15] = 1
binarized[binarized<.15] = 0

#save  binarized
np.save(f'{curr_dir}/fmri/pre_proc/{vid}_binary_face cov.npy', binarized) """