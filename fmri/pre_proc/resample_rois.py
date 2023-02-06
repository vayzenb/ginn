"""
resample ROIs to infant functional space
"""

import os
import shutil
import glob

from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn import image, plotting, datasets, masking
import nibabel as nib
import numpy as np
import pdb

study_dir = '/lab_data/behrmannlab/vlad/ginn'
data_dir = f'{study_dir}/fmri/aeronaut/preprocessed_standard/linear_alignment'
roi_dir = f'{study_dir}/fmri/aeronaut/derivatives/rois'

out_dir = f'{study_dir}/fmri/aeronaut/derivatives/rois'

sub = "mov_01_Z"

rois = ["mni_mask", "FFA", 'LOC', 'EVC','EAC']

#load data
bold_vol = image.load_img(f'{data_dir}/{sub}.nii.gz')

#extract first volume
first_vol = image.index_img(bold_vol, 0)


def resample_roi():
    #load mni mask
    mni_mask = load_mni152_brain_mask()
    mni_mask_resample = image.resample_to_img(mni_mask,first_vol, interpolation='nearest')
    #binarize mni mask
    mni_mask_resample = image.binarize_img(mni_mask_resample)


    #save mni mask
    nib.save(mni_mask_resample, f'{out_dir}/mni_mask.nii.gz')

    #for lr in ['l','r']:
    for roi in rois:
        #load roi
        roi_vol = image.load_img(f'{roi_dir}/{roi}.nii.gz')
        #resample roi to functional space
        resampled_roi = image.resample_to_img(roi_vol, first_vol, interpolation='nearest')
        #binarize roi
        resampled_roi = image.binarize_img(resampled_roi)

        #save resampled roi
        nib.save(resampled_roi, f'{out_dir}/{roi}.nii.gz')
        print(f'{roi} resampled')
    
def split_roi_hemis():
    
    #Make left/right versions of each
    for roi in rois:
        #load roi



        for lr in ['l','r']:
            roi_vol = image.load_img(f'{out_dir}/{roi}.nii.gz')
            affine = roi_vol.affine
            roi_data = image.get_data(roi_vol)
            #pdb.set_trace()
            
            mid = list((np.array((roi_data.shape))/2).astype(int)) #find mid point of image
            print(lr, mid, roi_data.shape)
            if lr == 'l':
                roi_data[:mid[0], :, :] = 0 

            elif lr == 'r':
                roi_data[mid[0]:, :, :] = 0 

            #convert back to nifti
            resampled_roi = nib.Nifti1Image(roi_data, affine)
            resampled_roi = image.binarize_img(resampled_roi)

            #save left and right rois
            nib.save(resampled_roi, f'{out_dir}/{lr}{roi}.nii.gz')
            
            #print(f'{roi} split into left and right')
            
split_roi_hemis()