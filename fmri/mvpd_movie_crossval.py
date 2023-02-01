'''
MVPD analyses where movie timecourse is split into training and testing sets
run from predict_srm_mvpd.py
'''

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
import brainiak.funcalign.srm
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
fix_tr = params.fix_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

file_suf = params.file_suf


subj_dir= data_dir


out_dir = f'{data_dir}/group_func'

roi_dir = f'{study_dir}/derivatives/rois'




rois = ['LO','FFA', 'A1']

ages = ['infant', 'adult']



features = [25]  # How many features will you fit?

n_iter = 30  # How many iterations of fitting will you perform

folds = 2

summary_cols = ['age', 'roi','corr']
suf = 'movie_crossval'

def extract_pc(data, n_components=None):

    """
    Extract principal components
    if n_components isn't set, it will extract all it can
    """
    
    pca = PCA(n_components = n_components)
    pca.fit(data)
    
    return pca

# %%
def calc_pc_n(pca, thresh):
    '''
    Calculate how many PCs are needed to explain X% of data
    
    pca - result of pca analysis
    thresh- threshold for how many components to keep
    '''
    
    explained_variance = pca.explained_variance_ratio_
    
    var = 0
    for n_comp, ev in enumerate(explained_variance):
        var += ev #add each PC's variance to current variance
        #print(n_comp, ev, var)

        if var >=thresh: #once variance > than thresh, stop
            break
    return n_comp+1


# %%

def extract_roi_data(curr_subs, roi):
    '''
    load subs into numpy array
    '''
    print(f'extracting {roi} data...')
    n = 0
    all_data = []
    for sub in curr_subs['participant_id']:
        whole_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/whole_brain_ts.npy')
        whole_ts = whole_ts[fix_tr:,:]

        #remove global signal
        if global_signal == 'pca':
            pca = extract_pc(whole_ts, n_components = 10)
            whole_confound = pca.transform(whole_ts)
        elif global_signal == 'mean':
            whole_confound = np.mean(whole_ts,axis =1)

        
        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}_ts_all.npy')
        sub_ts = sub_ts[fix_tr:,:]
        sub_ts = signal.clean(sub_ts,confounds = whole_confound, standardize_confounds=True)

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


def calc_mvpd(train_seed, test_seed, train_target, test_target):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """
    
    all_scores = []
    weights = []

    
    train_seed, test_seed, train_target, test_target = np.transpose(train_seed), np.transpose(test_seed), np.transpose(train_target), np.transpose(test_target)


    
    for kk in range(0,train_target.shape[1]):
        
        
        clf.fit(train_seed, train_target[:,kk])
        pred_ts = clf.predict(test_seed)
        
        
        corr =np.corrcoef(pred_ts,test_target[:,kk])[0,1]
        #weighted_corr = corr * ((train_target.shape[1]-kk)/train_target.shape[1])
        weighted_corr = corr
        weights.append(((train_target.shape[1]-kk)/train_target.shape[1]))

        


        all_scores.append(weighted_corr)

    
    final_score = np.sum(all_scores)/(np.sum(weights))
    #final_score = np.mean(all_scores)
    return final_score

def cross_val_srm(target_data,seed_data, n_feats):
    
    
    #roi_data = np.asanyarray(seed_data)
    target_data = np.asanyarray(target_data)
    seed_data = np.asanyarray(seed_data)
    seed_data = np.transpose(seed_data)

    # Create the SRM objects

    #srm_target.fit(roi_data)
    #srm_data = np.transpose(srm_target.s_)
    score = []

    #split into folds
    for fold in range(0,2):
        srm_target = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
        
        
        if fold == 0:
            #split data and run SRM on target
            train_target = target_data[:,:,0:int(target_data.shape[2]/2)]
            test_target = target_data[:,:,int(target_data.shape[2]/2):]
            
            
            srm_target.fit(train_target)
            train_target = srm_target.s_
            test_target = srm_target.transform(test_target)[0]

            train_seed = seed_data[:,0:int(seed_data.shape[1]/2)]
            test_seed = seed_data[:,int(seed_data.shape[1]/2):]
            


        elif fold ==1:
            #split data and run SRM on target
            train_target = target_data[:,:,int(target_data.shape[2]/2):]
            test_target = target_data[:,:,0:int(target_data.shape[2]/2)]

            srm_target.fit(train_target)
            train_target = srm_target.s_
            test_target = srm_target.transform(test_target)[0]

            train_seed = seed_data[:,int(seed_data.shape[1]/2):]
            test_seed = seed_data[:,0:int(seed_data.shape[1]/2)]

        curr_score = calc_mvpd(train_seed, test_seed, train_target, test_target)
        score.append(curr_score)

    return np.mean(score)

def predict_srm(seed_ts,n_feats=25):

    
    sub_summary = pd.DataFrame(columns = ['age','roi', 'corr'])
    for age in ages:
        if age == 'adult':
            curr_subs = sub_list[sub_list['Age'] >= 18]
        else:
            curr_subs = sub_list[sub_list['Age'] < 18]
        
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
                    
                
