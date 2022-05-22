"""
Downloads Healthy Brain Network (HBN) MRI Data

Checks whether it contains movie data

Removes unneeded files
"""
import os, shutil
import subprocess
import pandas as pd
import pdb
from glob import glob as glob

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn/fmri'
sub_list = pd.read_csv(f'{curr_dir}/HBN-Site-CBIC.csv')

root_url = 'https://fcp-indi.s3.amazonaws.com/data/Archives/HBN/MRI/Site-CBIC'
dest_dir = '/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn'

keep_files = ['*T1w.nii.gz','*T1w.json', '*movieDM_bold.nii.gz', '*movieDM_bold.json' ]
upper_age = 8

for sn, ss in enumerate(sub_list['participant_id']):
    
    #check if age is within range
    if sub_list['Age'].iloc[sn] < upper_age or sub_list['Age'].iloc[sn] >= 18: 
        
        if os.path.exists(f'{dest_dir}/sub-{ss}/') == False: #check if file was already downloaded
            try:
                print(f'{root_url}/{ss}.tar.gz')
                #download
                bash_cmd = f'wget -c {root_url}/sub-{ss}.tar.gz -P {dest_dir}'
                #bash_cmd = f'wget -c {root_url}/sub-{ss}.tar.gz'
                subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
                
                #extract tarball
                bash_cmd = f'tar -xzf {dest_dir}/sub-{ss}.tar.gz -C {dest_dir}'
                subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
                
                #remove tar
                os.remove(f'{dest_dir}/sub-{ss}.tar.gz')
                #pdb.set_trace()
                #Remove whole folder if the movie data isn't there
                if os.path.exists(f'{dest_dir}/sub-{ss}/func/sub-{ss}_task-movieDM_bold.nii.gz') == False:
                    bash_cmd = f'rm -rf {dest_dir}/sub-{ss}'
                    subprocess.run(bash_cmd.split())
                
                else: #otherwise remove unecessary files
                                        
                    #remove fieldmap folder
                    if os.path.exists(f'{dest_dir}/sub-{ss}/fmap'): shutil.rmtree(f'{dest_dir}/sub-{ss}/fmap')
                    if os.path.exists(f'{dest_dir}/sub-{ss}/dwi'): shutil.rmtree(f'{dest_dir}/sub-{ss}/dwi')
                    

                    #delete unnecessary files from anat
                    all_files = glob(f'{dest_dir}/sub-{ss}/anat/*')

                    for nifti_file in all_files:
                        if 'T1w'in nifti_file:
                            continue
                        else:
                            os.remove(nifti_file)

                    #delete unnecessary files from func
                    all_files = glob(f'{dest_dir}/sub-{ss}/func/*')

                    for nifti_file in all_files:
                        if 'movieDM' in nifti_file:
                            continue
                        else:
                            os.remove(nifti_file)
            except:
                print(f'Error with {ss}.tar.gz')



                
            
        
        
