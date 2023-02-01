"""Merges data by age group """
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import os, argparse
from glob import glob
import pdb
import pandas as pd
import numpy as np
from nilearn import image
import nibabel as nib
import ginn_params as params

parser = argparse.ArgumentParser(description='HBN preprocessing')
parser.add_argument('--age', required=True,
                    help='age of subjects to process', 
                    default=None)   

suf = ''

args = parser.parse_args()

age_groups = [int(args.age)]
age = 'adult'

print('libraries loaded')
#set up folders and ROIS
exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fix_tr = params.fix_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

#if adult extract subs over 18 #else extract subs under 18
if age == 'adult':
    sub_list = sub_list[sub_list['Age'] >= 18]
    sub_list = sub_list.reset_index(drop=True)
else:
    sub_list = sub_list[sub_list['Age'] < 18]
    sub_list = sub_list.reset_index(drop=True)

file_suf = params.file_suf

roi_dir=f'{study_dir}/derivatives/rois'
subj_dir=f'{study_dir}/derivatives'
data_dir = f'{study_dir}/Aeronaut_firstview/preprocessed_standard/linear_alignment/'

#whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
#whole_brain_mask = image.binarize_img(whole_brain_mask)

rois=["LO", "FFA", "A1"]
out_dir = f'{subj_dir}/mean_func'



whole_brain_mask = image.load_img(f'{subj_dir}/rois/mni_mask.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)

def get_existing_files(curr_subs):
    file_list =[]
    
    for sub in curr_subs:
        img = f'{subj_dir}/{sub}/{sub}_task-{file_suf}_bold.nii.gz'
        
        if os.path.exists(img):
            file_list.append(sub)

    return file_list




n =0
for age in age_groups:
    if age == '18':
        curr_subs = sub_list[sub_list['Age']>=age]
    else:
        curr_subs = sub_list[(sub_list['Age']>=age) & (sub_list['Age']<age+1)]

    curr_subs = curr_subs['participant_id']
    mean_file = f'{out_dir}/mean_task-{file_suf}_bold_{age}.nii.gz'
    
    file_list = get_existing_files(curr_subs)
    print(age, len(file_list))
    
    
    
    for sub in sub_list['participant_id']:
        print(f'Ages {age}:  {n+1} of {len(file_list)}')
        #pdb.set_trace()
        whole_brain_mask = image.binarize_img(image.load_img(f'{subj_dir}/{sub[1]}/{sub[1]}_analysis_mask.nii.gz'))
        affine = whole_brain_mask.affine

        bold_vol = image.load_img(f'{subj_dir}/{sub[1]}/{sub[1]}_task-{file_suf}_bold.nii.gz') #load data
        bold_vol= image.get_data(image.clean_img(bold_vol,standardize=True,mask_img=whole_brain_mask)) #standardize within mask and convert to numpy

        if n == 0:
            mean_img = np.zeros(bold_vol.shape)
            
        

        mean_img = mean_img +bold_vol

        n = n +1
        
    mean_img = mean_img/len(file_list)
    mean_img = nib.Nifti1Image(mean_img, affine)  # create the volume image
    nib.save(mean_img, mean_file)