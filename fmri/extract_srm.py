# %%
import warnings
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import os 
import pandas as pd

import numpy as np
from scipy import stats

from sklearn.decomposition import PCA

from brainiak.isc import isc
from brainiak.fcma.util import compute_correlation
import brainiak.funcalign.srm
from brainiak import image, io
from nilearn import signal

import matplotlib.pyplot as plt
import ginn_params as params
import pdb


# %%
global_signal = 'pca'
exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fix_tr = params.fix_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

file_suf = params.file_suf


out_dir = f'{data_dir}/group_func'
print(out_dir)


roi_dir = f'{study_dir}/derivatives/rois'


os.makedirs(out_dir, exist_ok=True)

rois = ['LO','FFA', 'A1']

age = 'infant'

features = [25,50,100,200]  # How many features will you fit?
features = [25]
n_iter = 30  # How many iterations of fitting will you perform

#if adult extract subs over 18 #else extract subs under 18
if age == 'adult':
    sub_list = sub_list[sub_list['Age'] >= 18]
    sub_list = sub_list.reset_index(drop=True)
else:
    sub_list = sub_list[sub_list['Age'] < 18]
    sub_list = sub_list.reset_index(drop=True)




#seed_ts = np.load(f'{subj_dir}/sub-{curr_subs["sub"][0]}/timeseries/seed_ts_all.npy')

# %%


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
    for sub in sub_list['participant_id']:
        whole_ts = np.load(f'{data_dir}/sub-{sub}/timeseries/whole_brain_ts.npy')
        whole_ts = whole_ts[fix_tr:,:]

        #remove global signal
        if global_signal == 'pca':
            pca = extract_pc(whole_ts, n_components = 10)
            whole_confound = pca.transform(whole_ts)
        elif global_signal == 'mean':
            whole_confound = np.mean(whole_ts,axis =1)

        
        sub_ts = np.load(f'{data_dir}/sub-{sub}/timeseries/{roi}_ts_all.npy')
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



for n_feats in features:
    
    # Create the SRM object
    srm = brainiak.funcalign.srm.SRM(n_iter=n_iter, features=n_feats)
    for rr in rois:
        for lr in ['l','r']:
            
            roi = f'{lr}{rr}'
            print(f'{roi} with {n_feats} features')
            
            roi_data = extract_roi_data(sub_list, roi)

            roi_data = standardize_data(roi_data)
            

            # Fit the SRM data
            print('Fitting SRM, may take a minute ...')
            srm.fit(roi_data)
            
            print('SRM has been fit')    
            


            #Save SRM
            np.save(f'{out_dir}/srm_{roi}_{age}_{n_feats}.npy',srm.s_)
