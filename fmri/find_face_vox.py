'''
Determines which voxels in each ROI are most responsive to faces

Uses a binary face cov which calculates mean responses to top 50% of faces in vid
'''


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
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

roi_dir=f'{study_dir}/derivatives/rois'


sub = sys.argv[1]


    
print(sub)

#whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
#whole_brain_mask = image.binarize_img(whole_brain_mask)

rois=["FFA", "EVC", "A1"]
roi_suf = '_face'

face_cov = np.load(f'{curr_dir}/fmri/pre_proc/{vid}_binary_face_cov.npy')
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
    

    #remove voxels cols with 0s
    roi_data = roi_data[:,~np.all(roi_data == 0, axis=0)]
    
    print('Seed data extracted...')

    return roi_data



sub_file = f'{data_dir}/{sub}/{sub}_task-movieDM_bold.nii.gz'
sub_file = f'{data_dir}/{sub}{file_suf}.nii.gz'

whole_brain_mask = image.binarize_img(image.load_img(f'{roi_dir}/mni_mask.nii.gz'))
print(sub_file)

def extract_face_vox(mv_ts, roi):

    mv_ts = mv_ts[fix_tr:,:]
    #integrate this into a funciton later
    #extract face voxels by multiplying by face cov
    face_resps = mv_ts*face_cov


    #calculate mean seperately for each column
    face_means = np.mean(face_resps, axis=0)

    #sort mv_ts in descending order by face_mean
    sort_idx = np.argsort(face_means)[::-1]
    sorted_ts = mv_ts[:,sort_idx]
    
    #extract first 10% of voxels
    top_10 = int(sorted_ts.shape[1]*.1)
    top_10_ts = sorted_ts[:,0:top_10]

    #standardize top 10% of voxels
    top_10_ts = (top_10_ts - np.mean(top_10_ts, axis=0)) / np.std(top_10_ts, axis=0)
    
    
    #save 
    np.save(f'{out_dir}/{roi}{roi_suf}',top_10_ts)

    

if os.path.exists(sub_file):
    print(f'Extracting for...{sub}')
    #grab  functional image in each sub dir
    

    out_dir = f'{subj_dir}/sub-{sub}/timeseries'
    os.makedirs(out_dir, exist_ok=True)
    
    bold_vol = image.load_img(sub_file) #load data


    #mean_ts = np.mean(whole_ts, axis=1)
    
    bold_vol = image.clean_img(bold_vol,standardize=False,mask_img=whole_brain_mask, detrend=True) #extract within brain mask

    for rr in rois:
        '''
        Extract mean TS
        '''
 
        '''
        Extract ts from all voxels
        '''

        #pdb.set_trace()
        #extract bilateral, left and right hemis
        mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/{rr}.nii.gz')
        extract_face_vox(mv_ts, rr)    

        mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/l{rr}.nii.gz')
        extract_face_vox(mv_ts , f'l{rr}')
        
        mv_ts = extract_mv_ts(bold_vol, f'{roi_dir}/r{rr}.nii.gz')
        extract_face_vox(mv_ts, f'r{rr}')
        
    
        
else:
    print(f'No file for {sub}')



        
    
           
        


