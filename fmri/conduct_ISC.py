'''
Calculate ISCs for infants adults and cross-age
'''
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')

import warnings
import os


warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb


from sklearn.model_selection import ShuffleSplit, LeaveOneOut

from scipy import stats
from nilearn import signal
import nibabel as nib
import ginn_params as params
import analysis_funcs as af
import random
print('libraries loaded')

# threshold for PCA
global_signal = 'mean'
use_pc_thresh = True

cv_type = 'shuffle'
folds = 24
split_size = .5


exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fix_tr = params.fmri_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

file_suf = params.file_suf

subj_dir= data_dir

out_dir = f'{data_dir}/group_func'
results_dir = f'{curr_dir}/results/isc'

roi_dir = f'{study_dir}/derivatives/rois'

rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']
roi_suf = '_ts_all'
ages = ['infant', 'adult']
file_suf = roi_suf

#within age ISCs
    #each roi to roi

#between age ISCs
    #each roi to roi

def loo_isc(seed_data, target_data):
    #create list of numbers to index subjects

    sub_idx = list(range(len(seed_data)))
    
    
    all_isc = []
    for ind in sub_idx:
        #extract data for current ind as test
        
        test_data = target_data[ind]
        #test_data = np.mean(test_data, axis=0) #average across subjects

        #create seed data from remaining subjects
        group_data = seed_data[~np.isin(sub_idx, ind)]
        #group_data = np.mean(group_data, axis=1) #average across voxels
        group_data = np.mean(group_data, axis=0)#average across subjects
        
        #calculate ISC
        isc = np.corrcoef(group_data, test_data)[0,1]

        all_isc.append(isc)
    
    return all_isc

def shuffle_isc(seed_data, target_data, folds = folds, split_size=split_size):
    #create list of numbers to index subjects

    sub_idx = list(range(len(seed_data)))
    
    
    all_isc = []
    for fold in range(0,folds):
        #shuffle list
        random.shuffle(sub_idx)
        
        test_data = target_data[sub_idx[:int(len(sub_idx)*split_size)],:]
        #test_data = np.mean(test_data, axis=1) #average across voxels
        test_data = np.mean(test_data, axis=0) #average across subjects

        group_data = seed_data[sub_idx[int(len(sub_idx)*split_size):],:]
        #group_data = np.mean(group_data, axis=1) #average across voxels
        group_data = np.mean(group_data, axis=0)#average across subjects
        
        #calculate ISC
        isc = np.corrcoef(group_data, test_data)[0,1]

        all_isc.append(isc)
    
    return all_isc


if cv_type == 'loo':
    cv = loo_isc

elif cv_type == 'shuffle':
    cv = shuffle_isc


summary_df = pd.DataFrame(columns=['seed_age', 'target_age', 'seed_roi', 'target_roi', 'age_cond', 'roi_cond', 'isc_mean', 'isc_se'])
#long nested loop to load data based on age and rois
for seed_age in ages:
    
    seed_subs = sub_list[sub_list['AgeGroup']==seed_age]

    for target_age in ages:
        target_subs = sub_list[sub_list['AgeGroup']==target_age]

        if seed_age == target_age:
            age_cond = 'within'
        else:
            age_cond = 'between'

        for seed_roi in rois:
            seed_data = af.extract_roi_data(subj_dir, seed_subs, seed_roi,roi_suf = roi_suf, fix_tr=fix_tr,global_signal = 'mean')
            
            #convert to numpy array
            seed_data = np.asarray(seed_data)
            
            #seed_data = np.mean(seed_data, axis=0) #average across subs
            for target_roi in rois:
                
                if seed_roi == target_roi:
                    roi_cond = 'within'
                else:
                    roi_cond = 'between'

                print(f'Calculating ISC for {seed_age} {seed_roi} to {target_age} {target_roi}')

                target_data = af.extract_roi_data(subj_dir, target_subs, target_roi,roi_suf = roi_suf, fix_tr=fix_tr,global_signal = 'mean')
                target_data = np.asarray(target_data)
                #target_data = np.mean(target_data, axis=0) #average across sub
                
                
                isc = cv(seed_data, target_data)

                isc_mean = np.mean(isc)
                isc_se = np.std(isc)
                curr_data = [seed_age, target_age, seed_roi, target_roi, age_cond, roi_cond, isc_mean, isc_se]

                #append to summary df
                summary_df.loc[len(summary_df)] = curr_data
            
            summary_df.to_csv(f'{results_dir}/isc_human_{cv_type}{file_suf}.csv', index=False)



        

    

                
                



