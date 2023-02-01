"""
Run MVPD on each individaul

Called from the predict_indiv_mvpd.py script
"""


curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}/fmri')
sys.path.insert(1, f'{curr_dir}')

import warnings
import os, argparse
from matplotlib.pyplot import subplot

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import ginn_params as params
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge

from nilearn import image, datasets
import nibabel as nib
import statsmodels.api as sm
print('libraries loaded')

# threshold for PCA

use_pc_thresh = True


pc_thresh = .9

clf = Ridge()
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

roi_dir = f'{study_dir}/derivatives/rois'
rois = ['LO','FFA','A1']
ages = [3,4,5,18]

#curr_subs= curr_subs[curr_subs['Age']<8]

#load whole brain mask
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)

summary_cols = ['sub','age', 'roi','corr']
suf = 'indiv'

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
    for n_components, ev in enumerate(explained_variance):
        var += ev #add each PC's variance to current variance
        #print(n_comp, ev, var)

        if var >=thresh: #once variance > than thresh, stop
            break
    return n_components+1


def calc_mvpd(train_seed,test_seed,train_target,test_target, target_pca):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """
    

    all_scores = []
    for pcn in range(0,len(target_pca.explained_variance_ratio_)):
        
        clf.fit(train_seed, train_target[:,pcn]) #fit seed PCs to target
        pred_ts = clf.predict(test_seed) #predict target PCs
        corr = np.corrcoef(pred_ts, test_target[:,pcn])[0,1] #calculate correlation
        #lm_model = sm.OLS(target_comps[:,pcn], seed_comps).fit()
        #r_squared = clf.score(seed_comps,target_comps[:,pcn]) 
        #r_squared = lm_model.rsquared

        weighted_corr = corr * target_pca.explained_variance_ratio_[pcn]
        all_scores.append(weighted_corr)
        #all_scores.append(r_squared)

    final_corr = np.sum(all_scores)/(np.sum(target_pca.explained_variance_ratio_))
    #final_corr = np.mean(all_scores)
    return final_corr


def cross_val(roi_data,seed_data):
    temp_corr = []
    for fold in range(0,2):
        
        if fold == 0:
            train_target = roi_data[0:int(roi_data.shape[0]/2),:]
            test_target = roi_data[int(roi_data.shape[0]/2):,:]

            train_seed = seed_data[0:int(seed_data.shape[0]/2),:]
            test_seed = seed_data[int(seed_data.shape[0]/2):,:]
            

        elif fold == 1:
            train_target = roi_data[int(roi_data.shape[0]/2):,:]
            test_target = roi_data[0:int(roi_data.shape[0]/2),:]

            train_seed = seed_data[int(seed_data.shape[0]/2):,:]
            test_seed = seed_data[0:int(seed_data.shape[0]/2),:]

        if use_pc_thresh == True: n_comp = calc_pc_n(extract_pc(train_target),pc_thresh) #determine number of PCs in train_data using threshold        
            
        pca = extract_pc(train_target, n_comp) #conduct PCA one more time with that number of PCs
        train_target = pca.transform(train_target) #transform train data in PCs    
        test_target = pca.transform(test_target) #transform test data in PCs

        
        corr = calc_mvpd(train_seed,test_seed, train_target, test_target,  pca)
        temp_corr.append(corr)
    
    return np.mean(temp_corr)


def predict_indvidual(seed_ts):
    """
    Calculate r2 of regression
    """


    
    sub_summary = pd.DataFrame(columns = ['sub','age','roi', 'corr'])


    #print(f'predicting for: {slr}{sr}', seed_comps.shape[1])
    for sub in enumerate(sub_list['participant_id']):
        #print(f'predicting {sub} from {args.roi}', seed_comps.shape[1], f'{sub[0]+1} of {len(sub_list)}')
        for roi in rois:
            for lr in ['l','r']:
                
                
                sub_ts = np.load(f'{subj_dir}/sub-{sub[1]}/timeseries/{lr}{roi}_ts_all.npy')
                sub_ts =sub_ts[fix_tr:,:]

                corr = cross_val(sub_ts,seed_ts)               
                
                

                curr_data = pd.Series([sub[1],sub_list['Age'][sub[0]], f'{lr}{roi}', corr],index= sub_summary.columns)
                sub_summary = sub_summary.append(curr_data,ignore_index = True)
                
                #pdb.set_trace()

    return sub_summary

            

                


