curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
from glob import glob as glob
import os

'''
Rename folders or files
'''
og_name = 'sub-sub-'
new_name = 'sub-'

target_dir = '/lab_data/behrmannlab/vlad/ginn/fmri/hbn/derivatives'

# get all files with the old video name
files = glob(f'{target_dir}/*{og_name}*')

# rename the files
for file in files:
    new_file = file.replace(og_name, new_name)
    os.rename(file, new_file)
    print(f'{file} renamed to {new_file}')

    

