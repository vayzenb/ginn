"""
Run RSA analysese
"""

import warnings
warnings.filterwarnings("ignore")
import os, argparse
from matplotlib.pyplot import subplot
from scipy import stats


import pandas as pd
import numpy as np
import pdb

human_predict = True
model_predict = True
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

"""Model params"""
vid = 'DM-clip'
stim_dir = f"{curr_dir}/stim/fmri_videos/frames"



model_archs = ['cornet_z_cl','cornet_z_sl']
train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer_types = ['aIT','pIT']

"""
fmri params
"""
exp = 'pixar'
#set directories


if exp == 'pixar':
    exp_dir= f'fmri/pixar'
    file_suf = 'pixar_run-001_swrf'
    all_subs = pd.read_csv(f'{curr_dir}/fmri/pixar-sub-info.csv')
    fix_tr = 6

elif exp == 'hbn':
    exp_dir = f'fmri/hbn'
    file_suf = 'movieDM'
    all_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')
    fix_tr = 0

raw_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/{exp_dir}'
study_dir = f'/lab_data/behrmannlab/vlad/ginn/'
out_dir = f'{study_dir}/{exp_dir}/derivatives/'
subj_dir=f'{raw_dir}/derivatives/preprocessed_data'
roi_dir = f'{study_dir}/derivatives/rois'

results_dir =f'{curr_dir}/results/rsa'

rois = ['LO','FFA','A1']
ages = [3,4,5,18]




def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/{sub[1]}/{sub[1]}_task-{file_suf}_bold.nii.gz'
        
        if os.path.exists(img):
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file

def create_mean_rdm():
    """
    Create mean rdm for each age and ROI
    """
    print('Creating mean rdms...')
    for age in ages:
        curr_subs = sub_list[sub_list['age'] == age]
        
        for lr in ['l','r']:
            for roi in rois:
                print(f'Creating mean rdm for {roi} {lr} {age}')
                all_subs = []
                for sub in curr_subs['sub']:
                    
                    curr_sub = pd.read_csv(f'{data_dir}/{sub}/derivatives/rdms/{lr}{roi}_rdm_vec.csv').to_numpy()
                    curr_sub = stats.zscore(curr_sub)
                    all_subs.append(curr_sub)

                
                all_subs = np.array(all_subs)
                
                
                mean_rdm = np.mean(all_subs, axis=0)
                
                np.savetxt(f'{out_dir}/group_func/rdm_{lr}{roi}_{age}_mean.csv', mean_rdm, delimiter=',')

'''
Get list of subs
'''

sub_list = get_existing_files(all_subs)
sub_list['age'] = sub_list['age'].apply(np.floor)
sub_list['age'][sub_list['age']>=18] = 18
sub_list = sub_list.drop_duplicates(subset ="sub",)
sub_list = sub_list.reset_index()





if human_predict == True:
    data_dir = f'/lab_data/behrmannlab/vlad/ginn/{exp_dir}/'

    

    create_mean_rdm()

    print('Running human RSA...')
    summary_df = pd.DataFrame(columns=['seed_age','seed_roi','target_age','target_roi','corr'])
    seed_age = 18
    for target_lr in ['l','r']:
        for target_roi in rois:
            for seed_lr in ['l','r']:
                for seed_roi in rois:
                    predictor_rdm = np.loadtxt(f'{out_dir}/group_func/rdm_{seed_lr}{seed_roi}_{seed_age}_mean.csv')
                        
                    
                    for target_age in ages:
                        if target_age != seed_age:
                            target_rdm = np.loadtxt(f'{out_dir}/group_func/rdm_{target_lr}{target_roi}_{target_age}_mean.csv', delimiter=',')
                            
                            
                            corr = np.corrcoef(predictor_rdm, target_rdm)[0,1]
                            summary_df = summary_df.append(pd.Series([seed_age, seed_lr+seed_roi, target_age, target_lr + target_roi, corr], index = summary_df.columns), ignore_index = True)
                        else:
                            continue

    summary_df.to_csv(f'{results_dir}/human_rsa_summary.csv', index=False)


        
if model_predict == True:
    print('Running model RSA...')
    data_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/rdms"

    for model_arch in model_archs:
        summary_df = pd.DataFrame(columns=['model_arc','train_type','layer','target_age','target_roi','corr'])
        for train_type in train_types:
            for layer in layer_types:
                print(f'Predicting using {model_arch} {train_type} {layer}')
                #load model rdm
                predictor_rdm = pd.read_csv(f'{data_dir}/{model_arch}_{train_type}_{layer}_{vid}_rdm_vec.csv', delimiter=',')
                predictor_rdm = np.ravel(predictor_rdm)
                
                for target_lr in ['l','r']:
                    for target_roi in rois:
                        for target_age in ages:
                            target_rdm = np.loadtxt(f'{results_dir}/{target_lr}{target_roi}_{target_age}_mean_rdm.csv', delimiter=',')
                            
                            corr = np.corrcoef(predictor_rdm, target_rdm)[0][1]
                            summary_df = summary_df.append(pd.Series([model_arch, train_type, layer, target_age, target_lr + target_roi, corr], index = summary_df.columns), ignore_index = True)


        summary_df.to_csv(f'{results_dir}/{model_arch}_rsa_summary.csv', index=False)
    #




