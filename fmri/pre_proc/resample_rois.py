"""
resample ROIs to infant functional space
"""

import os
import shutil
import glob

from nilearn.datasets import load_mni152_template, load_mni152_brain_mask
from nilearn import image, plotting, datasets, masking
import nibabel as nib

study_dir = '/lab_data/behrmannlab/vlad/ginn'
data_dir = f'{study_dir}/fmri/infant/Aeronaut_firstview/preprocessed_standard/linear_alignment'
roi_dir = f'{study_dir}/fmri/pixar/derivatives/rois'

out_dir = f'{study_dir}/fmri/infant/derivatives/rois'

sub = "mov_01_Z"

rois = ["LO", "FFA", "A1"]

#load data
bold_vol = image.load_img(f'{data_dir}/{sub}.nii.gz')

#extract first volume
first_vol = image.index_img(bold_vol, 0)

#load mni mask
mni_mask = load_mni152_brain_mask()
mni_mask_resample = image.resample_to_img(mni_mask,first_vol, interpolation='nearest')
#binarize mni mask
mni_mask_resample = image.binarize_img(mni_mask_resample)


#save mni mask
nib.save(mni_mask_resample, f'{out_dir}/mni_mask.nii.gz')

for lr in ['l','r']:
    for roi in rois:
        #load roi
        roi_vol = image.load_img(f'{roi_dir}/{lr}{roi}.nii.gz')
        #resample roi to functional space
        resampled_roi = image.resample_to_img(roi_vol, first_vol, interpolation='nearest')
        #binarize roi
        resampled_roi = image.binarize_img(resampled_roi)

        #save resampled roi
        nib.save(resampled_roi, f'{out_dir}/{lr}{roi}.nii.gz')
        print(f'{roi} resampled')
        