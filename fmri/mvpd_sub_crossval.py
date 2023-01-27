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
exp_dir= f'ginn/fmri/hbn'
study_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{study_dir}/derivatives/preprocessed_data'
out_dir = f'{study_dir}/derivatives/mean_func'
results_dir =f'{curr_dir}/results/mvpd'
roi_dir = f'{study_dir}/derivatives/rois'
all_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')

rois = ['LO','FFA', 'A1']
ages = [5,6,7,18]


features = [25,50,100,200]  # How many features will you fit?
n_iter = 30  # How many iterations of fitting will you perform

folds = 5
#curr_subs= curr_subs[curr_subs['Age']<8]

#load whole brain mask
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)

predictor_dir = '/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn/derivatives/group_func'
seed_ts = np.load(f'{predictor_dir}/srm_rLO_18_25.npy')                    
seed_ts = np.transpose(seed_ts)

def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/sub-{sub[1]}/sub-{sub[1]}_task-movieDM_bold.nii.gz'
        
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
        
        
        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}_ts_all.npy')
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

def cross_val_srm(roi_data,n_feats):
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
                
                score, score_se = cross_val_srm(roi_data,n_feats)
                
                
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
            

                


