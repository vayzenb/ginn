
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')

import numpy as np
import pandas as pd
import os
from glob import glob as glob
import ginn_params as params

import pdb

exp = 'hbn'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

sub_files = glob(f'{subj_dir}/*')

#remove directory path from sub_files
sub_files = [os.path.basename(x) for x in sub_files]

new_sub_list = pd.DataFrame(columns = sub_list.columns)

ages = [5, 6, 7]
n_subs = 24

new_sub_list = pd.DataFrame(columns = sub_list.columns)
for age in ages:
    #extract sub_list for age
    age_sub_list = sub_list[sub_list['AgeGroup'] == age]

    #randomize sub_list
    age_sub_list = age_sub_list.sample(frac=1).reset_index(drop=True)
    #pdb.set_trace()
    #remove first n_subs from sub_list
    keep_sub_list = age_sub_list.iloc[:n_subs]
    move_sub_list = age_sub_list.iloc[n_subs:]
    
    for sub in move_sub_list['participant_id'].values:
        #check if sub folder exists
        if os.path.exists(f'{subj_dir}/{sub}'):

            
            #move sub folder to unused folder
            os.rename(f'{subj_dir}/{sub}', f'{subj_dir}/unused/{sub}')

    #append sub to new sub_list
    new_sub_list = new_sub_list.append(keep_sub_list)

#append over 18 subs to new sub_list
new_sub_list = new_sub_list.append(sub_list[sub_list['AgeGroup'] == 18])

#save new sub_list
new_sub_list.to_csv(f'{curr_dir}/fmri/{exp}-sub-info.csv', index = False)


