"""
Run MVPD on each individaul
"""


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

from nilearn import image, datasets
import nibabel as nib
import statsmodels.api as sm
print('libraries loaded')

# threshold for PCA

use_pc_thresh = True


pc_thresh = .9

clf = Ridge()

exp = 'pixar'
#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

if exp == 'pixar':
    exp_dir= f'fmri/pixar'
    file_suf = 'pixar_run-001_swrf'
    all_subs = pd.read_csv(f'{curr_dir}/fmri/pixar-sub-info.csv')

elif exp == 'hbn':
    exp_dir = f'fmri/hbn'
    file_suf = 'movieDM'
    all_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')


raw_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/{exp_dir}'
study_dir = f'/lab_data/behrmannlab/vlad/ginn/'
out_dir = f'{study_dir}/{exp_dir}/derivatives/group_func'
subj_dir=f'{raw_dir}/derivatives/preprocessed_data'


roi_dir = f'{study_dir}/derivatives/rois'
rois = ['LO','FFA','A1']
ages = [3,4,5,18]

#curr_subs= curr_subs[curr_subs['Age']<8]

#load whole brain mask
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain_mask.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)


def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/{sub[1]}/{sub[1]}_task-{file_suf}_bold.nii.gz'
        
        if os.path.exists(img):
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file


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


def calc_mvpd(seed_data,train_target,test_target, target_pca):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs
    
    """
    train_seed = seed_data[0:int(seed_data.shape[0]/2),:]
    test_seed = seed_data[int(seed_data.shape[0]/2):,:]

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
            train_data = roi_data[0:int(roi_data.shape[0]/2),:]
            test_data = roi_data[int(roi_data.shape[0]/2):,:]
            
        elif fold == 1:
            train_data = roi_data[int(roi_data.shape[0]/2):,:]
            test_data = roi_data[0:int(roi_data.shape[0]/2),:]

        if use_pc_thresh == True: n_comp = calc_pc_n(extract_pc(train_data),pc_thresh) #determine number of PCs in train_data using threshold        
            
        pca = extract_pc(train_data, n_comp) #conduct PCA one more time with that number of PCs
        train_pcs = pca.transform(train_data) #transform train data in PCs    
        test_pcs = pca.transform(test_data) #transform test data in PCs

        temp_corr.append(calc_mvpd(seed_data, train_pcs, test_pcs, pca))
    
    return np.mean(temp_corr)


def calc_mvpd_r2(seed_ts):
    """
    Calculate r2 of regression
    """

    
    sub_list = get_existing_files(all_subs)
    sub_list = sub_list.reset_index()

    
    sub_summary = pd.DataFrame(columns = ['sub','age','roi', 'corr'])


    #print(f'predicting for: {slr}{sr}', seed_comps.shape[1])
    for sub in enumerate(sub_list['sub']):
        #print(f'predicting {sub} from {args.roi}', seed_comps.shape[1], f'{sub[0]+1} of {len(sub_list)}')
        for roi in rois:
            for lr in ['l','r']:
                
                
                sub_ts = np.load(f'{subj_dir}/{sub[1]}/timeseries/{lr}{roi}_ts_all.npy')

                corr = cross_val(sub_ts,seed_ts)               
                
                

                curr_data = pd.Series([sub[1],sub_list['age'][sub[0]], f'{lr}{roi}', corr],index= sub_summary.columns)
                sub_summary = sub_summary.append(curr_data,ignore_index = True)
                
                #pdb.set_trace()

    return sub_summary

            

                


