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

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn/'
sub_list = pd.read_csv(f'{curr_dir}/fmri/hbn-sub-info.csv')

root_url = 'https://fcp-indi.s3.amazonaws.com/data/Archives/HBN/MRI/Site-CBIC'
dest_dir = '/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn'

keep_files = ['*T1w.nii.gz','*T1w.json', '*movieDM_bold.nii.gz', '*movieDM_bold.json' ]
upper_age = 8


#list of ages to download
age_list = list(range(5,19))
age_list = [5]
n_age = 2 # number of subs to download for each age

#sort subs by AgeGroup
sub_list = sub_list.sort_values(by=['AgeGroup'])
pdb.set_trace()
#loop through age groups
for age in age_list:
    curr_subs = sub_list[sub_list['AgeGroup'] == age]

    n_count = 0
    #loop through subs and download
    for sn, ss in enumerate(curr_subs['participant_id']):
        
        
        if os.path.exists(f'{dest_dir}/{ss}/') == False: #check if file was already downloaded
            #try:
            print(f'{root_url}/{ss}.tar.gz')
            #download
            bash_cmd = f'wget -c {root_url}/{ss}.tar.gz -P {dest_dir}'
            #bash_cmd = f'wget -c {root_url}/sub-{ss}.tar.gz'
            subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
            
            #extract tarball
            bash_cmd = f'tar -xzf {dest_dir}/{ss}.tar.gz -C {dest_dir}'
            subprocess.run(bash_cmd.split(),check=True, capture_output=True, text=True)
            
            #remove tar
            os.remove(f'{dest_dir}/{ss}.tar.gz')
            #pdb.set_trace()
            #Remove whole folder if the movie data isn't there
            if os.path.exists(f'{dest_dir}/{ss}/func/{ss}_task-movieDM_bold.nii.gz') == False:
                bash_cmd = f'rm -rf {dest_dir}/{ss}'
                subprocess.run(bash_cmd.split())
            
            else: #otherwise remove unecessary files and keep DM movie
                                    
                #remove fieldmap folder
                if os.path.exists(f'{dest_dir}/{ss}/fmap'): shutil.rmtree(f'{dest_dir}/{ss}/fmap')
                if os.path.exists(f'{dest_dir}/{ss}/dwi'): shutil.rmtree(f'{dest_dir}/{ss}/dwi')
                

                #delete unnecessary files from anat
                all_files = glob(f'{dest_dir}/{ss}/anat/*')

                for nifti_file in all_files:
                    if 'T1w'in nifti_file:
                        continue
                    else:
                        os.remove(nifti_file)

                #delete unnecessary files from func
                all_files = glob(f'{dest_dir}/{ss}/func/*')

                for nifti_file in all_files:
                    if 'movieDM' in nifti_file:
                        continue
                    else:
                        os.remove(nifti_file)

                n_count += 1

            #except:
            #    print(f'Error with {ss}.tar.gz')

        #break out of loop if n_age subs have been downloaded
        if n_count == n_age:
            break


                
            
        
        
