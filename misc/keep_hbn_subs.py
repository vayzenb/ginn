
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
for sub in sub_files:
    #check if sub exists in sub_list
    if sub in sub_list['participant_id'].values:
        #append sub to new sub_list
        new_sub_list = new_sub_list.append(sub_list[sub_list['participant_id'] == sub])

#remove duplicates
new_sub_list = new_sub_list.drop_duplicates()
        


#save new sub_list
new_sub_list.to_csv(f'{curr_dir}/fmri/{exp}-sub-info.csv', index = False)