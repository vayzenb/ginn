curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
from glob import glob as glob
import os

'''
Rename timeseries files
'''
og_vid = 'Aeronaut'
new_vid = 'aeronaut'

data_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/rdms'

# get all files with the old video name
files = glob(f'{data_dir}/*{og_vid}*')

# rename the files
for file in files:
    new_file = file.replace(og_vid, new_vid)
    os.rename(file, new_file)
    print(f'{file} renamed to {new_file}')

    

