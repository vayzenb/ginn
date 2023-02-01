curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import pandas as pd
import numpy as np

exp = 'aeronaut'
vid = 'Aeronaut'

if exp == 'pixar':
    exp_dir= f'fmri/pixar'
    file_suf = 'pixar_run-001_swrf'
    sub_list = pd.read_csv(f'{curr_dir}/fmri/pixar-sub-info.csv')
    fix_tr = 6

elif exp == 'hbn':
    exp_dir = f'fmri/hbn'
    file_suf = 'movieDM'
    sub_list = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')
    fix_tr = 0

elif exp == 'aeronaut':
    exp_dir = f'fmri/aeronaut'
    file_suf = '_Z'
    sub_list = pd.read_csv(f'{curr_dir}/fmri/aeronaut-sub-info.csv')
    fix_tr = 3



if vid == 'DM-clip':
    '''
    Despicable me (DM) video params
    '''
    vols = 750 #volumes in the scan
    tr = .8 #TR of scan
    fix_tr =0 #number of throwaway volumes at beginning
    fps = 30 # frame per second of video (how many rows go into 1 sec)
    bin_size = int(fps * tr) # get the bin size to average by multiplying the FPS by tr

elif vid == 'partly_cloudy':

    '''
    Pixar video params
    '''
    vols = 168 #volumes in the scan
    tr = 2 #TR of scan
    fix_tr =0 #number of throwaway volumes
    fps = 24 # frame per second of video (how many rows go into 1 sec)
    bin_size = fps * tr # get the bin size to average by multiplying the FPS by tr

elif vid == 'Aeronaut':

    '''
    Pixar video params
    '''
    vols = 90 #volumes in the scan
    tr = 2 #TR of scan
    fix_tr =3 #number of throwaway volumes
    fps = 24 # frame per second of video (how many rows go into 1 sec)
    bin_size = fps * tr # get the bin size to average by multiplying the FPS by tr


study_dir = f'/lab_data/behrmannlab/vlad/ginn/{exp_dir}'
data_dir = f'{study_dir}/derivatives'
