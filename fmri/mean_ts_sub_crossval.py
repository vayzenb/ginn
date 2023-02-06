"""
Runs MVPD by fitting SRM of half of the data and then predicting SRM of other half
"""

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')

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

from scipy import stats
from nilearn import signal
import nibabel as nib
import ginn_params as params
import random
print('libraries loaded')

# threshold for PCA
global_signal = 'mean'
use_pc_thresh = True

pc_thresh = .9

clf = Ridge()

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fmri_tr = params.fmri_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

file_suf = params.file_suf
vols = params.vols

subj_dir= data_dir

out_dir = f'{data_dir}/group_func'

roi_dir = f'{study_dir}/derivatives/rois'

rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']

#suffix of roi to load
#options are _ts_all, _face, _nonface
roi_suf = '_nonface'
ages = ['infant', 'adult']


folds = 20


summary_cols = ['age', 'roi','corr']
suf = 'mean_sub_crossval'

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
        whole_ts = whole_ts[fmri_tr:,:]

        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}{roi_suf}.npy')
        

        if global_signal != '':
            #remove global signal
            if global_signal == 'pca':
                pca = extract_pc(whole_ts, n_components = 10)
                whole_confound = pca.transform(whole_ts)
            elif global_signal == 'mean':
                whole_confound = np.mean(whole_ts,axis =1)
            
            

            sub_ts = signal.clean(sub_ts,confounds = whole_confound, standardize_confounds=True)

        if sub_ts.shape[0] > vols:             
            sub_ts = sub_ts[fmri_tr:,:]

        sub_ts = np.transpose(sub_ts)
        #sub_ts = np.expand_dims(sub_ts,axis =2)
        
        sub_ts=np.mean(sub_ts, axis = 0)
        
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
        all_data[sub] = stats.zscore(all_data[sub], axis=0, ddof=1)
        all_data[sub] = np.nan_to_num(all_data[sub])
        
    return all_data


def fit_ts(seed_ts, train_data, test_data):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """

    
    all_scores = []
    clf = Ridge()
    
    clf.fit(seed_ts, train_data)
    pred_ts = clf.predict(seed_ts)

    score = np.corrcoef(pred_ts,test_data)[0,1]


       #all_scores.append(r_squared)

    #final_score = np.sum(all_scores)/(np.sum(target_pca.explained_variance_ratio_))
    
    return score

def cross_val(roi_data,seed_ts):
    print('running cross validation...')
    roi_data = np.asanyarray(roi_data)
    cv_ind = np.arange(0,len(roi_data)).tolist()
    
    score = []
    #split into folds
    for fold in range(0,folds):
        random.shuffle(cv_ind)
                
        train_ind = cv_ind[0:int(len(cv_ind)/2)]
        test_ind = cv_ind[int(len(cv_ind)/2):]
        #get training and testing data
        train_data = np.mean(roi_data[train_ind],0)
        test_data = np.mean(roi_data[test_ind],0)
        
        
        curr_score = fit_ts(seed_ts, train_data, test_data)
        score.append(curr_score)
    
    return np.mean(score), np.std(score)/np.sqrt(folds)

def predict_ts(seed_ts):

    
    sub_summary = pd.DataFrame(columns = ['age','roi', 'corr','se'])
    for age in ages:
        if age == 'adult':
            curr_subs = sub_list[sub_list['Age'] >= 18]
        else:
            curr_subs = sub_list[sub_list['Age'] < 18]

        
        for roi in rois:
        
            print(f'predicting {age} {roi} ...')
            
            #load all subject data from ROI
            roi_data = extract_roi_data(curr_subs, f'{roi}')
            #roi_data = standardize_data(roi_data)
            
            score, score_se = cross_val(roi_data,seed_ts)
            
            
            curr_data = pd.Series([age, f'{roi}', score, score_se],index= sub_summary.columns)
            sub_summary = sub_summary.append(curr_data,ignore_index = True)

                

    return sub_summary
                    
                


    


     

                


