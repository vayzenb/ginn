'''
Create summary file from different models
'''

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')

from scipy import stats
import pandas as pd
import numpy as np
import pdb

import ginn_params as params

exp = 'aeronaut'
analysis_type = 'mean_movie_crossval'


study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)
model_arch = 'cornet_z_sl'
train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']
file_suf = '_1'

results_dir = f'{curr_dir}/results/mean_ts'

#loop through models and layers
nn = 0
for model in train_types:
    for layer in layers:
        print(f'loading {exp}_{model_arch}_{model}_{layer}_{analysis_type}{file_suf}.csv')
        curr_df = pd.read_csv(f'{results_dir}/seperated/{exp}_{model_arch}_{model}_{layer}_{analysis_type}{file_suf}.csv')

        if nn == 0:
            summary_df = curr_df
        else:
            #add to summary df
            summary_df = pd.concat([summary_df, curr_df], axis=0)
        nn += 1
    
summary_df.to_csv(f'{results_dir}/{exp}_{model_arch}_{analysis_type}{file_suf}.csv', index=False)