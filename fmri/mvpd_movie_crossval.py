import warnings
import os, argparse
from matplotlib.pyplot import subplot

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge
import brainiak.funcalign.srm
from scipy import stats
from nilearn import image, datasets
import nibabel as nib
import random
print('libraries loaded')

# threshold for PCA

use_pc_thresh = True


pc_thresh = .9

clf = Ridge()

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

exp = 'pixar'
#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

if exp == 'pixar':
    exp_dir= f'fmri/pixar'
    file_suf = 'pixar_run-001_swrf'
    all_subs = pd.read_csv(f'{curr_dir}/fmri/pixar-sub-info.csv')
    fix_tr = 5

elif exp == 'hbn':
    exp_dir = f'fmri/hbn'
    file_suf = 'movieDM'
    all_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')
    fix_tr = 0

raw_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/{exp_dir}'
study_dir = f'/lab_data/behrmannlab/vlad/ginn/'
out_dir = f'{study_dir}/{exp_dir}/derivatives/group_func'
subj_dir=f'{raw_dir}/derivatives/preprocessed_data'
roi_dir = f'{study_dir}/derivatives/rois'




rois = ['LO','FFA', 'A1']

ages = [3,18]



features = [25,50,100,200]  # How many features will you fit?
n_iter = 30  # How many iterations of fitting will you perform

folds = 2




def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/{sub[1]}/{sub[1]}_task-{file_suf}_bold.nii.gz'
        
        if os.path.exists(img):
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file


# %%

def extract_roi_data(curr_subs, roi):
    '''
    load subs into numpy array
    '''
    print(f'extracting {roi} data...')
    n = 0
    all_data = []
    for sub in curr_subs['sub']:
        
        
        sub_ts = np.load(f'{subj_dir}/{sub}/timeseries/{roi}_ts_all.npy')
        
        sub_ts = sub_ts[fix_tr:,:]
        #sub_ts = np.transpose(sub_ts)
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


def calc_mvpd(train_seed, test_seed, train_data, test_data):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """
    #pdb.set_trace()
    all_scores = []

    for kk in range(0,train_data.shape[1]):
        clf.fit(train_seed, train_data[:,kk])
        pred_ts = clf.predict(test_seed)

        all_scores.append(np.corrcoef(pred_ts,test_data[:,kk])[0])


       #all_scores.append(r_squared)

    #final_score = np.sum(all_scores)/(np.sum(target_pca.explained_variance_ratio_))
    final_score = np.mean(all_scores)
    return final_score

def cross_val_srm(target_data,seed_data, n_feats):
    print('running cross validation...')
    roi_data = np.asanyarray(roi_data)
    
    # Create the SRM objects
    srm_target = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
    srm_seed = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
    srm_target.fit(roi_data)
    srm_data = np.transpose(srm.s_)
    score = []
    
    #split into folds
    for fold in range(0,2):
        
        if fold == 0:
            #split data and run SRM on target
            train_target = target_data[:,0:int(target_data.shape[1]/2)]
            test_target = target_data[:,int(target_data.shape[1]/2):]

            srm_target.fit(train_target)
            train_target = srm_target.s_
            test_target = srm_target.transform(test_target)

            train_seed = seed_data[:,0:int(seed_data.shape[1]/2)]
            test_seed = seed_data[:,int(seed_data.shape[1]/2):]

            #split data and run SRM on target
            srm_seed.fit(train_seed)
            train_seed = srm_seed.s_
            test_seed = srm_seed.transform(test_seed)

            srm_target = srm_seed.fit(target_data)
            train_srm = srm_target.s_
            test_srm = srm_target.transform(target_data)

        elif fold ==1:
            train_data = srm_data[int(srm_data.shape[0]/2):,:]
            test_data = srm_data[0:int(srm_data.shape[0]/2),:]

            train_seed = seed_data[int(seed_data.shape[0]/2):,:]
            test_seed = seed_data[0:int(seed_data.shape[0]/2),:]

        
        curr_score = calc_mvpd(train_seed, test_seed, train_data, test_data)
        score.append(curr_score)
    
    return np.mean(score)

def predict_srm(seed_ts,n_feats=50):

    
    sub_summary = pd.DataFrame(columns = ['age','roi', 'corr'])
    for age in ages:
        curr_subs = get_existing_files(all_subs)
        curr_subs['age'] = curr_subs['age'].apply(np.floor)
        curr_subs['age'][curr_subs['age']>=18] = 18
        curr_subs = curr_subs.drop_duplicates(subset ="sub",)
        curr_subs = curr_subs.reset_index()
        curr_subs = curr_subs[curr_subs['age']==age]
        
        
        for roi in rois:
            for lr in ['l','r']:
                print(f'predicting {age} {lr}{roi} ...')
                
                #load all subject data from ROI
                roi_data = extract_roi_data(curr_subs, f'{lr}{roi}')
                roi_data = standardize_data(roi_data)
                
                
                score = cross_val_srm(roi_data,seed_ts,n_feats)
                
                
                curr_data = pd.Series([age, f'{lr}{roi}', score],index= sub_summary.columns)
                sub_summary = sub_summary.append(curr_data,ignore_index = True)
    
                

    return sub_summary
                    
                

