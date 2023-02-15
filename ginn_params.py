curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import pandas as pd
import numpy as np

def load_params(exp):
    study_dir = f'/lab_data/behrmannlab/vlad/ginn/fmri/{exp}'
    sub_list = pd.read_csv(f'{curr_dir}/fmri/{exp}-sub-info.csv')

    if exp == 'pixar':
        vid = 'partly_cloudy'
        file_suf = 'pixar_run-001_swrf'
        
        fix_tr = 6

        data_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/fmri/pixar/derivatives/preprocessed_data'
        subj_dir = f'{study_dir}/derivatives'
        vols = 168 #volumes in the scan
        tr = 2 #TR of scan
        
        fps = 24 # frame per second of video (how many rows go into 1 sec)


    elif exp == 'hbn':
        '''
        Despicable me (DM) fmri and modelling params
        '''
        
        file_suf = '_task-movieDM_bold'
        
        fix_tr = 0
        
        data_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn/derivatives/preprocessed_data'
        subj_dir = f'{study_dir}/derivatives'
        
        vid = 'DM-clip'
        vols = 750 #volumes in the scan
        tr = .8 #TR of scan
        fix_tr =0 #number of throwaway volumes at beginning
        fps = 30 # frame per second of video (how many rows go into 1 sec)

        ages = [5, 6, 7, 18]
        #ages= [18]


    elif exp == 'aeronaut':
        vid = 'aeronaut'
        file_suf = '_Z'
        
        fix_tr = 3
        
        data_dir = f'{study_dir}/preprocessed_standard/linear_alignment'
        subj_dir = f'{study_dir}/derivatives'

        vols = 90 #volumes in the scan
        tr = 2 #TR of scan
        
        fps = 24 # frame per second of video (how many rows go into 1 sec)

        ages = ['adult', 'infant'] #age groups in the study
        
    bin_size = int(fps * tr) # get the bin size to average by multiplying the FPS by tr

    return study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages








