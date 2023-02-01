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

folds = 10


summary_cols = ['age', 'roi','corr']
suf = 'sub_crossval'

def extract_roi_data(curr_subs, roi):
    '''
    load subs into numpy array
    '''
    print(f'extracting {roi} data...')
    n = 0
    all_data = []
    for sub in curr_subs['participant_id']:
        
        
        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}_ts_all.npy')
        sub_ts = sub_ts[fix_tr:,:]
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


def calc_mvpd(seed_ts, train_data, test_data):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """

    
    all_scores = []

    for kk in range(0,train_data.shape[0]):
        clf.fit(seed_ts, train_data[kk,:])
        pred_ts = clf.predict(seed_ts)

        all_scores.append(np.corrcoef(pred_ts,test_data[kk,:])[0,1])


       #all_scores.append(r_squared)

    #final_score = np.sum(all_scores)/(np.sum(target_pca.explained_variance_ratio_))
    final_score = np.mean(all_scores)
    return final_score

def cross_val_srm(roi_data,seed_ts,n_feats):
    print('running cross validation...')
    roi_data = np.asanyarray(roi_data)
    cv_ind = np.arange(0,len(roi_data)).tolist()
    
    score = []
    #split into folds
    for fold in range(0,folds):
        random.shuffle(cv_ind)
        
        # Create the SRM objects
        train_srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
        test_srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
        
        train_ind = cv_ind[0:int(len(cv_ind)/2)]
        test_ind = cv_ind[int(len(cv_ind)/2):]
        #get training and testing data
        train_data = list(roi_data[train_ind])
        test_data = list(roi_data[test_ind])
        
        train_srm.fit(train_data)
        test_srm.fit(test_data)
        curr_score = calc_mvpd(seed_ts, train_srm.s_, test_srm.s_)
        score.append(curr_score)
    
    return np.mean(score), np.std(score)/np.sqrt(folds)

def predict_srm(seed_ts,n_feats=25):

    
    sub_summary = pd.DataFrame(columns = ['age','roi', 'corr','se'])
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
                
                score, score_se = cross_val_srm(roi_data,seed_ts,n_feats)
                
                
                curr_data = pd.Series([age, f'{lr}{roi}', score, score_se],index= sub_summary.columns)
                sub_summary = sub_summary.append(curr_data,ignore_index = True)
    
                

    return sub_summary
                    
                


    


    
    

    


"""
def predict_srm(seed_ts,n_comp =10):
    
    Calculate r2 of regression
    
    
    sub_list = get_existing_files(curr_subs)
    sub_list = sub_list.drop_duplicates()
    sub_list = sub_list.reset_index()
    
    
    sub_summary = pd.DataFrame(columns = ['sub','age','roi', 'r2'])

    #print(f'predicting for: {slr}{sr}', seed_comps.shape[1])
    for sub in enumerate(sub_list['sub']):
        #print(f'predicting {sub} from {args.roi}', seed_comps.shape[1], f'{sub[0]+1} of {len(sub_list)}')
        for roi in rois:
            for lr in ['l','r']:
                
                sub_ts = np.load(f'{subj_dir}/sub-{sub[1]}/timeseries/{lr}{roi}_ts_all.npy')
                
                
                if use_pc_thresh == True: n_comp = calc_pc_n(extract_pc(sub_ts),pc_thresh) #determine number of PCs in train_data using threshold        
                
                child_pca = extract_pc(sub_ts, n_comp) #conduct PCA one more time with that number of PCs
                child_comps = child_pca.transform(sub_ts) #transform train data in PCs       
                         

                r2 = calc_mvpd(seed_ts,child_comps, child_pca)
                
                

                curr_data = pd.Series([sub[1],sub_list['age'][sub[0]], f'{lr}{roi}', r2],index= sub_summary.columns)
                sub_summary = sub_summary.append(curr_data,ignore_index = True)
                

    return sub_summary
"""
            

                


