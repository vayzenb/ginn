
from nilearn import image, plotting
import nibabel as nib
import numpy as np
import pandas as pd
import pdb

rois = ['LO','FFA', 'A1']

roi_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn/derivatives/rois'
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
affine = whole_brain_mask.affine

for lr in ['l','r']:
    for target_roi in rois:
        print(f'{lr}{target_roi}')
        
        curr_roi = image.get_data(image.binarize_img(image.load_img(f'{roi_dir}/full/{lr}{target_roi}.nii.gz')))
        
        for sub_roi in rois:
            if sub_roi != target_roi:
                print(sub_roi)
                new_roi = image.get_data(image.binarize_img(image.load_img(f'{roi_dir}/full/{lr}{sub_roi}.nii.gz')))
            
                curr_roi = curr_roi - new_roi
        curr_roi[curr_roi >0] = 1
        curr_roi[curr_roi <1] = 0
        
        curr_roi = nib.Nifti1Image(curr_roi, affine)
        nib.save(curr_roi, f'{roi_dir}/{lr}{target_roi}.nii.gz')
        print(f'{roi_dir}/{lr}{target_roi}.nii.gz')
