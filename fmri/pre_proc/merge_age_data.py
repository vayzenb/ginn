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

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn/fmri'
exp_dir= f'ginn/fmri/hbn'
study_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{study_dir}/derivatives/preprocessed_data'
out_dir = f'{study_dir}/derivatives/mean_func'
sub_list = pd.read_csv(f'{curr_dir}/HBN-Site-CBIC.csv')

whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)

def get_existing_files(curr_subs):
    file_list =[]
    for sub in curr_subs:
        img = f'{subj_dir}/sub-{sub}/sub-{sub}_task-movieDM_bold.nii.gz'
        
        if os.path.exists(img):
            file_list.append(img)

    return file_list




n =0
for age in age_groups:
    if age == 18:
        curr_subs = sub_list[sub_list['Age']>=age]
    else:
        curr_subs = sub_list[(sub_list['Age']>=age) & (sub_list['Age']<age+1)]

    curr_subs = curr_subs['participant_id']
    mean_file = f'{out_dir}/mean_task-movieDM_bold_{age}.nii.gz'
    
    file_list = get_existing_files(curr_subs)
    print(age, len(file_list))
    
    
    for sub in enumerate(file_list):
        print(f'Ages {age}:  {n+1} of {len(file_list)}')

        bold_vol = image.load_img(sub[1]) #load data
        bold_vol= image.get_data(image.clean_img(bold_vol,standardize=True,mask_img=whole_brain_mask)) #standardize within mask and convert to numpy

        if n == 0:
            mean_img = np.zeros(bold_vol.shape)
            
        

        mean_img = mean_img +bold_vol

        n = n +1
        
    mean_img = mean_img/len(file_list)
    mean_img = nib.Nifti1Image(mean_img, affine)  # create the volume image
    nib.save(mean_img, mean_file)