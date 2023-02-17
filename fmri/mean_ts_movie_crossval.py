"""
Runs MVPD by fitting to half the movie data for subs and predicting other half
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

clf = LinearRegression()

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'





rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']
rois = ['LOC','FFA','A1','EVC']
#suffix of roi to load
#options are _ts_all, _face, _nonface
roi_suf = '_ts_all'


n_subs = 24
folds = 24


summary_cols = ['age','roi', 'corr','se']
suf = 'mean_movie_crossval'

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
        #check if sub has sub- in front, else add it
        if sub[0:4] != 'sub-':
            sub = f'sub-{sub}'


        #check if file exists
        if os.path.exists(f'{subj_dir}/{sub}/timeseries/{roi}{roi_suf}.npy'):
            whole_ts = np.load(f'{subj_dir}/{sub}/timeseries/whole_brain_ts.npy')
            whole_ts = whole_ts[fix_tr:,:]

            sub_ts = np.load(f'{subj_dir}/{sub}/timeseries/{roi}{roi_suf}.npy')
            
            if sub_ts.shape[0] > vols:             
                sub_ts = sub_ts[fix_tr:,:]
            

            if global_signal != '':
                #remove global signal
                if global_signal == 'pca':
                    pca = extract_pc(whole_ts, n_components = 10)
                    whole_confound = pca.transform(whole_ts)
                elif global_signal == 'mean':
                    whole_confound = np.mean(whole_ts,axis =1)
                
                

                sub_ts = signal.clean(sub_ts,confounds = whole_confound, standardize_confounds=True)

            

            sub_ts = np.transpose(sub_ts)
            #sub_ts = np.expand_dims(sub_ts,axis =2)
            
            sub_ts=np.mean(sub_ts, axis = 0)
            
            all_data.append(sub_ts)
    #pdb.set_trace()    
    return all_data

def tune_hyperparams(roi_data,seed_ts):



def fit_ts(seed_train,seed_test, train_data, test_data):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """
    #standardize all variables
    seed_train = stats.zscore(seed_train, axis=0)
    seed_test = stats.zscore(seed_test, axis=0)
    train_data = stats.zscore(train_data, axis=0)
    test_data = stats.zscore(test_data, axis=0)


    clf = Ridge()
    
    
    clf.fit(seed_train, train_data)
    pred_ts = clf.predict(seed_test)

    score = np.corrcoef(pred_ts,test_data)[0,1]

    
    return score

def cross_val(roi_data,seed_ts):
    
    print('running cross validation...')
    roi_data = np.asanyarray(roi_data)
    cv_ind = np.arange(0,len(roi_data)).tolist()

    roi_mean = np.mean(roi_data, axis = 0)
    mean_score_list = []
    #calcualte mean across movie halves for all subs
    #this is happening inside the fold loop so account for variability in ridge fitting
    for movie_half in range(0,2):
    
        if movie_half == 0:
            seed_train = seed_ts[0:int(len(seed_ts)/2),:]
            seed_test = seed_ts[int(len(seed_ts)/2):,:]
            target_train = roi_mean[0:int(len(roi_mean)/2)]
            target_test = roi_mean[int(len(roi_mean)/2):]
        elif movie_half == 1:
            seed_train = seed_ts[int(len(seed_ts)/2):,:]
            seed_test = seed_ts[0:int(len(seed_ts)/2),:]
            target_train = roi_mean[int(len(roi_mean)/2):]
            target_test = roi_mean[0:int(len(roi_mean)/2)]

        curr_score = fit_ts(seed_train,seed_test, target_train, target_test)
        mean_score_list.append(curr_score)

    mean_score = np.mean(mean_score_list)
    
    #calculate resampled SE

    
    score = []
    #split into folds
    for fold in range(0,folds):

        
        #sample cv_ind with replacement
        curr_ind = np.random.choice(cv_ind, len(cv_ind), replace = True)
        
        curr_data = np.mean(roi_data[curr_ind,:],axis = 0)

        for movie_half in range(0,2):
            
            if movie_half == 0:
                seed_train = seed_ts[0:int(len(seed_ts)/2),:]
                seed_test = seed_ts[int(len(seed_ts)/2):,:]
                target_train = curr_data[0:int(len(curr_data)/2)]
                target_test = curr_data[int(len(curr_data)/2):]
            elif movie_half == 1:
                seed_train = seed_ts[int(len(seed_ts)/2):,:]
                seed_test = seed_ts[0:int(len(seed_ts)/2),:]
                target_train = curr_data[int(len(curr_data)/2):]
                target_test = curr_data[0:int(len(curr_data)/2)]


            curr_score = fit_ts(seed_train,seed_test, target_train, target_test)
            score.append(curr_score)

    
    
    se = np.std(score)/np.sqrt(folds)
    return mean_score, se

def predict_ts(seed_ts, exp):
    global study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages, roi_dir
    study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)
    roi_dir = f'{study_dir}/derivatives/rois'
    
    sub_summary = pd.DataFrame(columns = ['age','roi', 'corr','se'])
    for age in ages:
        curr_subs = sub_list[sub_list['AgeGroup'] == age]
        #select first 24 subs in each age group
        curr_subs = curr_subs.head(n_subs)
        
        
        for roi in rois:
        
            print(f'predicting {age} {roi} ...')
            
            #load all subject data from ROI
            roi_data = extract_roi_data(curr_subs, f'{roi}')
            
            
            score, score_se = cross_val(roi_data,seed_ts)
            
            
            curr_data = pd.Series([age, f'{roi}', score, score_se],index= sub_summary.columns)
            sub_summary = sub_summary.append(curr_data,ignore_index = True)

                

    return sub_summary
                    
                


    


     

                


