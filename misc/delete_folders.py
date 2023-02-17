curr_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn'
import sys
sys.path.insert(1, f'{curr_dir}')

import os
import shutil
from glob import glob as glob

import numpy as np
import pandas as pd
import ginn_params as params
import pdb

exp = 'hbn'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

#loop throughs subs and delete timeseries data
for sub in sub_list['participant_id']:
    #check if file exists
    #pdb.set_trace()
    if os.path.exists(f'{subj_dir}/{sub}/timeseries'):
        print(f'deleting {sub} timeseries')
        shutil.rmtree(f'{subj_dir}/{sub}/timeseries')
        