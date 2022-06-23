"""
Run RSA analysese
"""

import warnings
warnings.filterwarnings("ignore")
import os, argparse
from matplotlib.pyplot import subplot


import pandas as pd
import numpy as np
import pdb

human_predict = False
model_predict = True
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

"""Model params"""
vid = 'DM-clip'
stim_dir = f"{curr_dir}/stim/fmri_videos/frames"

data_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/rdms"

model_archs = ['cornet_z_cl','cornet_z_sl']
train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer_types = ['aIT','pIT']

"""HBN params
#set directories



data_dir = f'/lab_data/behrmannlab/vlad/{exp_dir}/'
"""
exp_dir= f'ginn/fmri/hbn'
raw_data_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{raw_data_dir}/derivatives/preprocessed_data'
results_dir =f'{curr_dir}/results/rsa'
curr_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')
rois = ['LO','FFA','OFA']
ages = [18, 5,6,7]





def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/sub-{sub[1]}/sub-{sub[1]}_task-movieDM_bold.nii.gz'
        
        if os.path.exists(img):
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file

def create_mean_rdm():
    """
    Create mean rdm for each age and ROI
    """

    for age in ages:
        curr_subs = sub_list[sub_list['age'] == age]
        for lr in ['l','r']:
            for roi in rois:
                all_subs = []
                for sub in curr_subs['sub']:
                    all_subs.append(pd.read_csv(f'{data_dir}/sub-{sub}/derivatives/rdms/{lr}{roi}_rdm.csv').to_numpy())

                
                all_subs = np.array(all_subs)
                mean_rdm = np.mean(all_subs, axis=0)
                np.savetxt(f'{results_dir}/{lr}{roi}_{age}_mean_rdm.csv', mean_rdm, delimiter=',')

'''
Get list of subs
'''
sub_list = get_existing_files(curr_subs)
sub_list = sub_list.drop_duplicates()
sub_list = sub_list.reset_index()
#round ages to nearest group
sub_list['age'][sub_list['age'] >=18] = 18

sub_list['age'] = sub_list['age'].astype(float).round(0)

if human_predict == True:



    sub_list = get_existing_files(curr_subs)
    sub_list = sub_list.drop_duplicates()
    sub_list = sub_list.reset_index()
    summary_df = pd.DataFrame(columns=['seed_age','seed_roi','target_age','target_roi','corr'])
    seed_age = 18
    for target_lr in ['l','r']:
        for target_roi in rois:
            for seed_lr in ['l','r']:
                for seed_roi in rois:
                    predictor_rdm = np.loadtxt(f'{results_dir}/{seed_lr}{seed_roi}_{seed_age}_mean_rdm.csv', delimiter=',')
                    for target_age in ages:
                        if target_age != seed_age:
                            target_rdm = np.loadtxt(f'{results_dir}/{target_lr}{target_roi}_{target_age}_mean_rdm.csv', delimiter=',')
                            
                            corr = np.corrcoef(predictor_rdm, target_rdm)[0][1]
                            summary_df = summary_df.append(pd.Series([seed_age, seed_lr+seed_roi, target_age, target_lr + target_roi, corr], index = summary_df.columns), ignore_index = True)
                        else:
                            continue

    summary_df.to_csv(f'{results_dir}/human_rsa_summary.csv', index=False)


        
if model_predict == True:
    print('Running model RSA...')


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
    #create_mean_rdm()




