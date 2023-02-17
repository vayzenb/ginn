'''
Extract maximally and minamlly face-response voxels from each ROI
'''

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')

import warnings
import os


warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from nilearn import signal
import pdb

import ginn_params as params
import analysis_funcs as af

# threshold for PCA
global_signal = 'mean'
use_pc_thresh = True

cv_type = 'shuffle'
folds = 24
split_size = .5

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

out_dir = f'{subj_dir}/group_func'
#results_dir = f'{curr_dir}/results/isc'

roi_dir = f'{study_dir}/derivatives/rois'

rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']
#rois = ['rFFA']

face_cov = np.load(f'{curr_dir}/fmri/pre_proc/{exp}_parametric_face_cov.npy')

for sub in sub_list['participant_id']:
    sub_dir = f'{subj_dir}/sub-{sub}/timeseries'


    for roi in rois:
        print(f'{sub} {roi}')
        #load roi data
        roi_data = np.load(f'{sub_dir}/{roi}_ts_all.npy')
        roi_data =np.loadtxt(f'{sub_dir}/{roi}_ts_mean.txt')
        roi_data = roi_data[fix_tr:,:]

        #remove voxels cols with 0s
        roi_data = roi_data[:,~np.all(roi_data == 0, axis=0)]

        #correlate each voxel to face cov
        roi_corr = np.zeros(roi_data.shape[1])
        for v in range(roi_data.shape[1]):
            roi_corr[v] = np.corrcoef(roi_data[:,v],face_cov)[0,1]

        #get top 10% of voxels
        top_10 = np.argsort(roi_corr,axis=0)[-int(roi_data.shape[1]*0.1):]
        #extract top 10% of voxels
        top_10_roi_data = roi_data[:,top_10]
        
        #save mean of top 10% of voxels
        np.save(f'{sub_dir}/{roi}_face.npy',top_10_roi_data)

        #get bottom 10% of voxels
        bottom_10 = np.argsort(roi_corr,axis=0)[:int(roi_data.shape[1]*0.1)]
        #extract bottom 10% of voxels
        bottom_10_roi_data = roi_data[:,bottom_10]
        
        #save mean of bottom 10% of voxels
        np.save(f'{sub_dir}/{roi}_nonface.npy',bottom_10_roi_data)
        
        #save roi_correlations
        np.save(f'{sub_dir}/{roi}_roi_corr.npy',roi_corr)
        








