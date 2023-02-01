"""Merges data by age group """
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import os, argparse
from glob import glob
import pdb
import pandas as pd
import numpy as np
from nilearn import image
import nibabel as nib
from nilearn import signal
import ginn_params as params


age = 'adult'
global_signal = 'mean'
print('libraries loaded')
#set up folders and ROIS
exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fix_tr = params.fix_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

ages = ['infant', 'adult']

file_suf = params.file_suf

roi_dir=f'{study_dir}/derivatives/rois'
subj_dir=f'{study_dir}/derivatives'
data_dir = f'{study_dir}/Aeronaut_firstview/preprocessed_standard/linear_alignment/'

#whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
#whole_brain_mask = image.binarize_img(whole_brain_mask)

rois=["LO", "FFA", "A1"]
out_dir = f'{subj_dir}/group_func'

#create output directory
os.makedirs(out_dir, exist_ok=True)

def extract_pc(data, n_components=None):

    """
    Extract principal components
    if n_components isn't set, it will extract all it can
    """
    
    pca = PCA(n_components = n_components)
    pca.fit(data)
    
    return pca

def extract_roi_data(curr_subs, roi):
    '''
    load subs into numpy array
    '''
    print(f'extracting {roi} data...')
    n = 0
    all_data = []
    for sub in curr_subs['participant_id']:
        whole_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/whole_brain_ts.npy')
        

        #remove global signal
        if global_signal == 'pca':
            pca = extract_pc(whole_ts, n_components = 10)
            whole_confound = pca.transform(whole_ts)
        elif global_signal == 'mean':
            whole_confound = np.mean(whole_ts,axis =1)
        
        
        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}_ts_all.npy')
        
        sub_ts = signal.clean(sub_ts,confounds = whole_confound, standardize_confounds=True)
        
        
        all_data.append(sub_ts)
        #sub_ts = np.reshape(sub_ts, [sub_ts.shape[1], sub_ts.shape[0], sub_ts.shape[2]])
        #pdb.set_trace()
        '''
        if n == 0:
            all_data = sub_ts
            
            n += 1
        else:
            
            all_data = np.concatenate((all_data,sub_ts), axis = 2)
        '''
    return all_data


for age in ages:
        if age == 'adult':
            curr_subs = sub_list[sub_list['Age'] >= 18]
        else:
            curr_subs = sub_list[sub_list['Age'] < 18]

        
        for roi in rois:
            for lr in ['l','r']:
                
                
                #load all subject data from ROI
                roi_data = extract_roi_data(curr_subs, f'{lr}{roi}')

                
                #average roi_data
                roi_data = np.mean(roi_data, axis = 0)

                #save data
                np.save(f'{out_dir}/mean_{lr}{roi}_{age}_ts.npy', roi_data)


