"""
iteratively creates sbatch scripts to run multiple fmri jobs at once 
"""

import subprocess
from glob import glob as glob
import os
import time
import pdb
import time

mem = 12
run_time = "3-00:00:00"

#stim info
data_dir = f'/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn'
study_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn/fmri'
suf = '_preproc'
iter = 10 #how many jobs to submit before waiting
sleep_time = 45 # set how many minutes to wait between submissions

subj_list = [os.path.basename(x) for x in glob(f'{data_dir}/*')] #get list of subs to loop over

def setup_sbatch(sub):
    """
    Text for batch script
    """
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={sub}{suf}

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
#SBATCH --output={study_dir}/slurm_out/{sub}{suf}.out

module load fsl-6.0.3

python preprocess.py --path {data_dir} --subj {ss}
"""
    return sbatch_setup


n = 0
for ss in subj_list:

    job_name = f'{ss}_{suf}'
    print(job_name)
    
    #write job 
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(ss))
    
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True) #run job
    os.remove(f"{job_name}.sh") #delete sbatch script

    #iterate the number of jobs submitted
    #sleep when number is reached
    n = n + 1
    if n == iter:
        time.sleep(60*sleep_time)
        n = 0
