"""Merges data by age group """

import os, argparse
from glob import glob
import pdb
import pandas as pd
import numpy as np
from nilearn import image
import nibabel as nib

parser = argparse.ArgumentParser(description='HBN preprocessing')
parser.add_argument('--age', required=True,
                    help='age of subjects to process', 
                    default=None)   

suf = ''

args = parser.parse_args()

age_groups = [int(args.age)]

exp = 'pixar'
#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

if exp == 'pixar':
    exp_dir= f'fmri/pixar'
    file_suf = 'pixar_run-001_swrf'
    sub_list = pd.read_csv(f'{curr_dir}/fmri/pixar-sub-info.csv')

elif exp == 'hbn':
    exp_dir = f'fmri/hbn'
    file_suf = 'movieDM'
    sub_list = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')

exp_dir= f'ginn/fmri/pixar'
study_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{study_dir}/derivatives/preprocessed_data'
out_dir = f'/lab_data/behrmannlab/vlad/ginn/derivatives/mean_func'



whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
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
    if age == 18:
        curr_subs = sub_list[sub_list['Age']>=age]
    else:
        curr_subs = sub_list[(sub_list['Age']>=age) & (sub_list['Age']<age+1)]

    curr_subs = curr_subs['participant_id']
    mean_file = f'{out_dir}/mean_task-{file_suf}_bold_{age}.nii.gz'
    
    file_list = get_existing_files(curr_subs)
    print(age, len(file_list))
    
    
    
    for sub in enumerate(file_list):
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