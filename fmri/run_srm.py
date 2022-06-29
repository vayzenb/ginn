# %%
import warnings
import sys 
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import pandas as pd

import numpy as np
from scipy import stats
import scipy.spatial.distance as sp_distance
from sklearn.svm import NuSVC
import nibabel as nib

from brainiak.isc import isc
from brainiak.fcma.util import compute_correlation
import brainiak.funcalign.srm
from brainiak import image, io

import matplotlib.pyplot as plt
import pdb


# %%
#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
exp_dir= f'ginn/fmri/hbn'
study_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{study_dir}/derivatives/preprocessed_data'
out_dir = f'{study_dir}/derivatives/group_func'
results_dir =f'{curr_dir}/results/mvpd'
roi_dir = f'{study_dir}/derivatives/rois'

curr_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')

rois = ['LO','FFA', 'OFA']

age = 18

features = [25,50,100,200]  # How many features will you fit?
n_iter = 30  # How many iterations of fitting will you perform


# %%

def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/sub-{sub[1]}/sub-{sub[1]}_task-movieDM_bold.nii.gz'
        
        if os.path.exists(img):
            
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file

curr_subs = get_existing_files(curr_subs)
curr_subs = curr_subs[curr_subs['age']>=18]
curr_subs = curr_subs.drop_duplicates(subset ="sub",)
curr_subs = curr_subs.reset_index()


# %%

def extract_roi_data(curr_subs, roi):
    '''
    load subs into numpy array
    '''
    print(f'extracting {roi} data...')
    n = 0
    all_data = []
    for sub in curr_subs['sub']:
        
        
        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}_ts_all.npy')
        sub_ts = np.transpose(sub_ts)
        #sub_ts = np.expand_dims(sub_ts,axis =2)
        
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


def standardize_data(all_data):
    '''
    standardize data
    '''
    print('standardizing data...')
    
    for sub in range(0,len(all_data)):    
        

        # zscore each sub
        all_data[sub] = stats.zscore(all_data[sub], axis=1, ddof=1)
        all_data[sub] = np.nan_to_num(all_data[sub])
        
    return all_data

for n_feats in features:
    
    # Create the SRM object
    srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
    for rr in rois:
        for lr in ['l','r']:
            
            roi = f'{lr}{rr}'
            print(f'{roi} with {n_feats} features')
            
            roi_data = extract_roi_data(curr_subs, roi)

            roi_data = standardize_data(roi_data)
            pdb.set_trace()

            # Fit the SRM data
            print('Fitting SRM, may take a minute ...')
            srm.fit(roi_data)

            print('SRM has been fit')    
            


            #Save SRM
            np.save(f'{out_dir}/srm_{roi}_{age}_{n_feats}.npy',srm.s_)
