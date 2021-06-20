import subprocess
import os
from glob import glob

#import pdb

#set up folders and ROIS
exp_dir= 'ginn/preschool_fmri/derivatives'
study_dir = f'/lab_data/behrmannlab/vlad/{exp_dir}'
roi_dir=f'{study_dir}/ROIs'
subj_dir=f'{study_dir}/preprocessed_data'
out_dir = f'{study_dir}/timeseries'
ROIs=["LO", "PFS"]

#func to pull each subject directory
def load_files(dirName):
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(dirName):
        listOfFiles += dirnames
            
    return listOfFiles 

#pull sub dirs
subs = load_files(subj_dir)

#loop through subs
for ss in subs:
    #gran  functional image in each sub dir
    sub_file = glob(f'{subj_dir}/{ss}/*_swrf_bold.nii.gz')[0]
    
    for rr in ROIs:
        # create fsl command for left hemi ROI
        bash_cmd = f'fslmeants -i {sub_file} -o {out_dir}/{ss}_l{rr}_timecourse.txt -m {roi_dir}/l{rr}.nii.gz'

        #execute fsl command
        bash_out = subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)

        #same for right hemi ROI
        bash_cmd = f'fslmeants -i {sub_file} -o {out_dir}/{ss}_r{rr}_timecourse.txt -m {roi_dir}/r{rr}.nii.gz'

        #execute fsl command
        bash_out = subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
           
        


