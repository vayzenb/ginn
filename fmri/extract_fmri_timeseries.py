curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import subprocess
import os
from glob import glob
import pdb
from nilearn import image, maskers
import nibabel as nib
import numpy as np
import ginn_params as params
import pdb

print('libraries loaded')
#set up folders and ROIS
exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fix_tr = params.fix_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

file_suf = params.file_suf

roi_dir=f'{study_dir}/derivatives/rois'
subj_dir=f'{study_dir}/derivatives'
data_dir = f'{study_dir}/Aeronaut_firstview/preprocessed_standard/linear_alignment/'

#whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
#whole_brain_mask = image.binarize_img(whole_brain_mask)

rois=["LO", "FFA", "A1"]


#pull sub dirs
#subj_list = [os.path.basename(x) for x in glob(f'{data_dir}/*.nii.gz')] #get list of subs to loop over



def extract_mv_ts(bold_vol, mask_dir):
    """
    extract multivariate time course from ROI
    """

    #load seed
    roi = image.binarize_img(image.load_img(f'{mask_dir}'))
    masker = maskers.NiftiMasker(mask_img=roi)
    masker.fit(bold_vol)
    roi_data = masker.transform(bold_vol)
    
    print('Seed data extracted...')

    return roi_data


n = 1
#loop through subs
for sub in sub_list['participant_id']:
    
    
    #sub_file = f'{subj_dir}/{ss}/{ss}_task-movieDM_bold.nii.gz'
    sub_file = f'{data_dir}/{sub}{file_suf}.nii.gz'
    
    whole_brain_mask = image.binarize_img(image.load_img(f'{subj_dir}/rois/mni_mask.nii.gz'))

    if os.path.exists(sub_file):
        print(f'Extracting for...{sub}', f'{n} of {len(sub_list)}')
        #grab  functional image in each sub dir
        

        out_dir = f'{subj_dir}/sub-{sub}/timeseries'
        os.makedirs(out_dir, exist_ok=True)
        
        bold_vol = image.load_img(sub_file) #load data
        whole_masker = maskers.NiftiMasker(mask_img=whole_brain_mask, detrend = True, standardize = True)
        whole_masker.fit(bold_vol)
        whole_ts = whole_masker.transform(bold_vol)
        np.save(f'{out_dir}/whole_brain_ts',whole_ts)
        #mean_ts = np.mean(whole_ts, axis=1)
        
        bold_vol = image.clean_img(bold_vol,standardize=True,mask_img=whole_brain_mask, detrend=True) #extract within brain mask

        for rr in rois:
            '''
            Extract mean TS
            '''
            # create fsl command for left hemi ROI
            bash_cmd = f'fslmeants -i {sub_file} -o {out_dir}/l{rr}_ts_mean.txt -m {roi_dir}/l{rr}.nii.gz'
            #execute fsl command
            bash_out = subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)

            #same for right hemi ROI
            bash_cmd = f'fslmeants -i {sub_file} -o {out_dir}/r{rr}_ts_mean.txt -m {roi_dir}/r{rr}.nii.gz'
            #execute fsl command
            bash_out = subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)

            '''
            Extract ts from all voxels
            '''

            #pdb.set_trace()

            mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/l{rr}.nii.gz')
            np.save(f'{out_dir}/l{rr}_ts_all',mv_ts)
            mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/r{rr}.nii.gz')
            np.save(f'{out_dir}/r{rr}_ts_all',mv_ts)
        
            
    else:
        print(f'No file for {sub}')

    n = n +1

        
    
           
        


