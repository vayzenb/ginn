"""
Pre-processes hbn
"""
import os, shutil
import subprocess
import pandas as pd
import pdb
from glob import glob as glob
import os, argparse, shutil

parser = argparse.ArgumentParser(description='HBN preprocessing')
parser.add_argument('--path', required=True,
                    help='path to datadirect', 
                    default=None)   
parser.add_argument('--subj', required=True,
                    help='path to subject folder', 
                    default=None)

                 

args = parser.parse_args()

data_dir = args.path
sub = args.subj

#deskull anat
bash_cmd = f'bet {data_dir}/{sub}/anat/{sub}_acq-VNavNorm_T1w.nii.gz {data_dir}/{sub}/anat/{sub}_acq-VNavNorm_T1w_brain.nii.gz -R -B'
subprocess.run(bash_cmd.split())

#create motion spikes for func
bash_cmd = f"fsl_motion_outliers -i {data_dir}/{sub}/func/{sub}_task-movieDM_bold.nii.gz -o {data_dir}/{sub}/func/{sub}_task-movieDM_bold_spikes.txt --dummy=0"
subprocess.run(bash_cmd.split())

