"""
Run RSA analysese
"""
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')
import warnings
warnings.filterwarnings("ignore")
import os, argparse
from matplotlib.pyplot import subplot
from scipy import stats

import ginn_params as params
import pandas as pd
import numpy as np
import pdb

human_predict = True
model_predict = True
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

"""Model params"""

stim_dir = f"{curr_dir}/stim/fmri_videos/frames"



model_archs = ['cornet_z_sl']
train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer_types = ['aIT']

"""
fmri params
"""

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
vid = params.vid

subj_dir= data_dir

out_dir = f'{data_dir}/group_func'

roi_dir = f'{study_dir}/derivatives/rois'
results_dir = f'{curr_dir}/results/rsa'

rois = ['LO','FFA', 'A1']
ages = ['infant', 'adult']
seed_age = 'adult'

iter = 100

def bootstrap_se(predictor, target, iter):
    """
    Calculate standard error across bootstraps
    """
    corr_list = []
    for ii in range(0,iter):

        idx = np.random.choice(range(0,len(predictor)), len(predictor))
        corr = np.corrcoef(predictor[idx], target[idx])[0,1]
        corr_list.append(corr)

    return np.std(corr_list)


if human_predict == True:
    data_dir = f'/lab_data/behrmannlab/vlad/ginn/{exp_dir}/'



    print('Running human RSA...')
    summary_df = pd.DataFrame(columns=['age','roi','corr','seed_age','seed_roi', 'se'])
    
    for target_lr in ['l','r']:
        for target_roi in rois:
            for seed_lr in ['l','r']:
                for seed_roi in rois:
                    predictor_rdm = np.load(f'{out_dir}/{seed_lr}{seed_roi}_{seed_age}_rdm.npy')
                        
                    
                    for target_age in ages:
                        if target_age != seed_age:
                            target_rdm = np.load(f'{out_dir}/{target_lr}{target_roi}_{target_age}_rdm.npy')
                            
                            
                            corr = np.corrcoef(predictor_rdm, target_rdm)[0,1]
                            se = bootstrap_se(predictor_rdm, target_rdm, iter)
                            summary_df = summary_df.append(pd.Series([target_age, target_lr+target_roi,corr, seed_age, seed_lr+seed_roi, se], index = summary_df.columns), ignore_index = True)
                        else:
                            continue

    summary_df.to_csv(f'{results_dir}/{exp}_human_rsa.csv', index=False)


        
if model_predict == True:
    print('Running model RSA...')
    data_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/rdms"

    for model_arch in model_archs:
        summary_df = pd.DataFrame(columns=['age','roi','corr','architecture', 'train_type', 'layer','se'])
        for train_type in train_types:
            for layer in layer_types:
                print(f'Predicting using {model_arch} {train_type} {layer}')
                #load model rdm
                predictor_rdm = np.load(f'{data_dir}/{model_arch}_{train_type}_{layer}_{vid}_rdm.npy')
                
                
                for target_lr in ['l','r']:
                    for target_roi in rois:
                        for target_age in ages:
                            target_rdm = np.load(f'{out_dir}/{target_lr}{target_roi}_{target_age}_rdm.npy')
                            
                            corr = np.corrcoef(predictor_rdm, target_rdm)[0][1]
                            se = bootstrap_se(predictor_rdm, target_rdm, iter)
                            summary_df = summary_df.append(pd.Series([target_age, target_lr+target_roi,corr,model_arch, train_type, layer, se], index = summary_df.columns), ignore_index = True)


        summary_df.to_csv(f'{results_dir}/{exp}_model_rsa.csv', index=False)
    #




