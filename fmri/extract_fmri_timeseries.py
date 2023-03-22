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

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, fmri_suf, start_trs,end_trs, data_dir, vols, tr, fps, bin_size, ages= params.load_params(exp)

roi_dir=f'{study_dir}/derivatives/rois'

#read sub from arg
sub = sys.argv[1]

rois=["LOC","FFA", "EVC", "A1"]



def extract_mv_ts(bold_vol, mask_dir):
    """
    extract multivariate time course from ROI
    """

    #load seed
    roi = image.binarize_img(image.load_img(f'{mask_dir}'))
    masker = maskers.NiftiMasker(mask_img=roi)
    masker.fit(bold_vol)
    roi_data = masker.transform(bold_vol)
    
    #remove voxels cols with 0s
    roi_data = roi_data[:,~np.all(roi_data == 0, axis=0)]
    
    print('Seed data extracted...')

    return roi_data

def remove_confound_vols(mv_ts,confound):
    """
    remove confound vols from mv_ts
    """
    
    #replace confound vols with nans
    mv_ts[confound==1,:] = np.nan
    

    return mv_ts 

sub_file = f'{data_dir}/{sub}/{sub}_task-movieDM_bold.nii.gz'
sub_file = f'{data_dir}/{sub}{fmri_suf}.nii.gz'

whole_brain_mask = image.binarize_img(image.load_img(f'{roi_dir}/gm_mask.nii.gz'))
#print(sub_file)

if os.path.exists(sub_file):
    print(f'Extracting for...{sub}')

    
    #load eye confounds
    eye_conf = np.loadtxt(f'{study_dir}/eye_confounds/{sub}.txt').astype(int)
    #add two rows of zeros to eye confounds 
    #add two zeros to beginning of eye confounds
    eye_conf = np.pad(eye_conf,(2,0), 'constant')
    #trim last two rows of eye confounds
    eye_conf = eye_conf[:-2]

    #load motion confounds
    motion_conf = np.loadtxt(f'{study_dir}/motion_confounds/{sub}.txt').astype(int)
    #sum all rows to make motion confounds a vector
    
    if len(motion_conf.shape) > 1:
        motion_conf = np.sum(motion_conf, axis=1)
    
    #add two rows of zeros to motion confounds
    motion_conf = np.pad(motion_conf,(2,0), 'constant')
    #trim last two rows of motion confounds
    motion_conf = motion_conf[:-2]
    

    #grab  functional image in each sub dir
    out_dir = f'{subj_dir}/sub-{sub}/timeseries'
    os.makedirs(out_dir, exist_ok=True)
    
    bold_vol = image.load_img(sub_file) #load data
    whole_masker = maskers.NiftiMasker(mask_img=whole_brain_mask, detrend = True, standardize = True)
    whole_masker.fit(bold_vol)
    whole_ts = whole_masker.transform(bold_vol)
    
    whole_confound = np.mean(whole_ts,axis =1) #calc mean to later regress out global signal 

    #append motion confound to whole confound
    whole_confound = np.vstack((whole_confound,motion_conf))
    whole_confound = np.transpose(whole_confound)
    
    
    np.save(f'{out_dir}/whole_brain_ts',whole_ts)
    #mean_ts = np.mean(whole_ts, axis=1)
    
    #standardize values and regress out global signal
    bold_vol = image.clean_img(bold_vol,standardize=True,mask_img=whole_brain_mask, detrend=True, confounds = whole_confound) #extract within brain mask

    for rr in rois:

        '''
        Extract ts from all voxels
        '''

        #extract bilateral, left and right hemis
        mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/{rr}.nii.gz')
        mv_ts = remove_confound_vols(mv_ts, eye_conf)
        np.save(f'{out_dir}/{rr}',mv_ts)

        
        #extract left
        mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/l{rr}.nii.gz')
        mv_ts = remove_confound_vols(mv_ts, eye_conf)
        np.save(f'{out_dir}/l{rr}',mv_ts)

        #extract right
        mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/r{rr}.nii.gz')
        mv_ts = remove_confound_vols(mv_ts, eye_conf)
        np.save(f'{out_dir}/r{rr}',mv_ts)
    
        
else:
    print(f'No file for {sub}')



        
    
           
        


