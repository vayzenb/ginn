import os, argparse, shutil
import subprocess
from glob import glob as glob

#stim info
data_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn'
study_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn/fmri'


subj_list = [os.path.basename(x) for x in glob(f'{data_dir}/*')] #get list of subs to loop over
n = 1
for ss in subj_list:
    if os.path.exists(f'{data_dir}/{ss}/func/{ss}_task-movieDM_bold.nii.gz') == False:
        bash_cmd = f'rm -rf {data_dir}/{ss}'
        subprocess.run(bash_cmd.split())
        print('Removed all of:', ss)
    
    else:
        
    
        if os.path.exists(f'{data_dir}/{ss}/derivatives/fsl/1stLevel.feat/filtered_func_data.nii.gz') == False:
            bash_cmd = f'rm -rf {data_dir}/{ss}/derivatives/fsl/1stLevel.feat'
            subprocess.run(bash_cmd.split())
            print('Removed incomplete 1stlevel of:', ss)


        #Remove other spurious files
        #Remove whole folder if the movie data isn't there

        
        #remove fieldmap folder
        if os.path.exists(f'{data_dir}/{ss}/fmap'): shutil.rmtree(f'{data_dir}/{ss}/fmap')
        if os.path.exists(f'{data_dir}/{ss}/dwi'): shutil.rmtree(f'{data_dir}/{ss}/dwi')
        

        #delete unnecessary files from anat
        all_files = glob(f'{data_dir}/{ss}/anat/*')

        for nifti_file in all_files:
            if 'T1w'in nifti_file:
                continue
            else:
                os.remove(nifti_file)

        #delete unnecessary files from func
        all_files = glob(f'{data_dir}/{ss}/func/*')

        for nifti_file in all_files:
            if 'movieDM' in nifti_file:
                continue
            else:
                os.remove(nifti_file)


    print(f'{n} of {len(subj_list)}')

    n+= 1


