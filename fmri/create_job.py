"""
iteratively creates sbatch scripts to run multiple fmri jobs at once 
"""

import subprocess
from glob import glob as glob
import os
import time
import pdb
import time

mem = 24
run_time = "1-00:00:00"

#stim info
data_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn'
study_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn'
out_dir = f'{data_dir}/derivatives/preprocessed_data/'
anat = '/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz'
suf = '_reg'
iter = 12 #how many jobs to submit before waiting
sleep_time = 15 # set how many minutes to wait between submissions

subj_list = [os.path.basename(x) for x in glob(f'{data_dir}/*')] #get list of subs to loop over

rois = ['lLO','rLO', 'lOFA', 'rOFA','lFFA', 'rFFA']
ages = ['18','5','6','7']
print('Running: ', suf)

def setup_sbatch(params):
    """
    Text for batch script
    """
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={'_'.join(params)}{suf}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p cpu

#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0

# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Exclude
#SBATCH --exclude=mind-1-26,mind-1-30

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{'_'.join(params)}{suf}.out

module load fsl-6.0.3
conda activate fmri_new

# python pre_proc/preprocess.py --path {data_dir} --subj {params[0]}
python pre_proc/run_1stlevel.py --path {data_dir} --og_sub sub-NDARAB514MAJ --curr_sub {params[0]}
# python pre_proc/1stlevel2standard.py --anat {anat} --path {data_dir} --sub {params[0]}


"""
    return sbatch_setup

# python fmri/child_mvpd_r2.py --roi {params[0]} --age {params[1]}

n = 0
total_n = 1
'''
for age in ages:
    for roi in rois:
        params = [roi,age]
        job_name = f"{'_'.join(params)}{suf}"
        print(job_name , f'{total_n} out of {len(subj_list)}')
        
        #write job 
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(params))
        
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True) #run job
        os.remove(f"{job_name}.sh") #delete sbatch script

        #iterate the number of jobs submitted
        #sleep when number is reached
        n = n + 1
        
        if n == iter:
            time.sleep(60*sleep_time)
            n = 0

    total_n = total_n + 1
'''

for ss in subj_list:
    
    params = [ss]
    
    if os.path.exists(f'{out_dir}/{ss}/{ss}_task-movieDM_bold.nii.gz') == False:
        job_name = f"{'_'.join(params)}{suf}"
        print(job_name , f'{total_n} out of {len(subj_list)}')

        
        #write job 
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(params))
        
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True) #run job
        os.remove(f"{job_name}.sh") #delete sbatch script

        #iterate the number of jobs submitted
        #sleep when number is reached
        n = n + 1
        
        if n == iter:
            time.sleep(60*sleep_time)
            n = 0

    total_n = total_n + 1

