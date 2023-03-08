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

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)


#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

results_dir = f'{curr_dir}/results'

rois = ['FFA','A1','EVC'] + ['lFFA','lA1','lEVC'] + ['rFFA','rA1','rEVC']


#suffix of roi to load
#options are _ts_all, _face, _nonface
roi_suf = ''

if roi_suf == '':
    roi_suf = '_ts_all'

alpha = .05
n_subs = 24
folds = 100


summary_cols = ['age','roi', 'corr','se', 'ci_low','ci_high', 'isc']
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
    
    return all_data


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

def find_optimal_pc(roi_data, predictor_ts):
    
    if predictor_ts.shape[1] < vols:
        iter_num = 1
    else:
        iter_num = int(predictor_ts.shape[1]/vols)

    
    roi_mean = np.mean(roi_data,axis=0)
    
    
    all_scores = []
    for pc in range(iter_num,vols,iter_num):
        
        seed_ts = predictor_ts[:,0:pc]
        
        score = []
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

            curr_score = fit_ts(seed_train,seed_test,target_train,target_test)
            
            score.append(curr_score)
        all_scores.append(np.mean(score))
    
    #find index of max value
    max_ind = np.argmax(all_scores)
    
    
    
    return max_ind+1, all_scores[max_ind]

def cross_val(roi_data,predictor_ts):
    
    print('running cross validation...')
    roi_data = np.asanyarray(roi_data)
    cv_ind = np.arange(0,len(roi_data)).tolist()
    
    isc_score = []
    #split into folds
    score = []
    for fold in range(0,folds):

        #shuffle cv_ind use fold as the same random seed
        random.Random(fold).shuffle(cv_ind)
        
        #split into train and test for hyperparameter tuning
        #use train data to find how many PCs to use
        hp_train = roi_data[cv_ind[0:int(len(cv_ind)/2)]]
        hp_test = roi_data[cv_ind[int(len(cv_ind)/2):]]

        #find optimal PCs
        optimal_pc, _ = find_optimal_pc(hp_train, predictor_ts)

        seed_ts = predictor_ts[:,0:optimal_pc]

        #split into train and test for hyperparameter tuning
        hp_mean = np.mean(hp_train,axis=0)

        roi_mean = np.mean(hp_test,axis=0)

        #correlate hyperparameter test data with hyperparameter train data
        isc_score.append(np.corrcoef(hp_mean,roi_mean)[0,1])
        

        split_score = []
        for movie_half in range(0,2):
            
            if movie_half == 0:
                seed_train = seed_ts[0:int(len(seed_ts)/2),:]
                seed_test = seed_ts[int(len(seed_ts)/2):,:]
                target_train = roi_mean[0:int(len(roi_mean)/2)]
                target_test = roi_mean[int(len(roi_mean)/2):]

                hp_train = hp_train[0:int(len(hp_train)/2),:]
                hp_test = hp_test[0:int(len(hp_test)/2),:]
            elif movie_half == 1:
                seed_train = seed_ts[int(len(seed_ts)/2):,:]
                seed_test = seed_ts[0:int(len(seed_ts)/2),:]
                target_train = roi_mean[int(len(roi_mean)/2):]
                target_test = roi_mean[0:int(len(roi_mean)/2)]

                hp_train = hp_train[int(len(hp_train)/2):,:]
                hp_test = hp_test[int(len(hp_test)/2):,:]

            
            curr_score = fit_ts(seed_train,seed_test, target_train, target_test)

            split_score.append(curr_score)

        score.append(np.mean(split_score))
        

    mean_score = np.mean(score)
    #caluclate 95% confidence interval
    ci_low = np.percentile(score, alpha*100)
    ci_high= np.percentile(score, 100-alpha*100)

    se = np.std(score)/np.sqrt(folds)

    #convert score to np
    score = np.asanyarray(score)
    #save score

    isc = np.mean(isc_score)
    


    return mean_score, se, ci_low, ci_high, score, isc

def predict_ts(seed_ts, exp):
    global study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages, roi_dir
    study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)
    roi_dir = f'{study_dir}/derivatives/rois'
    
    
    sub_summary = pd.DataFrame(columns = summary_cols)
    boot_summary = pd.DataFrame()
    for age in ages:
        curr_subs = sub_list[sub_list['AgeGroup'] == age]
        #select first 24 subs in each age group
        curr_subs = curr_subs.head(n_subs)
        
        
        for roi in rois:
        
            print(f'predicting {age} {roi} ...')
            
            
            #load all subject data from ROI
            roi_data = extract_roi_data(curr_subs, f'{roi}')
            
            
            
            score, score_se,ci_low, ci_high, scores, isc = cross_val(roi_data,seed_ts)

            boot_summary[f'{age}_{roi}'] = scores


            
            
            curr_data = pd.Series([age, f'{roi}', score, score_se,ci_low, ci_high, isc],index= sub_summary.columns)
            sub_summary = sub_summary.append(curr_data,ignore_index = True)

            

                

    return sub_summary, boot_summary
                    
                


    


     

                


