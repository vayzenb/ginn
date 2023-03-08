"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import subprocess
from glob import glob
import os
import time
import pdb
import ginn_params as params

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, fmri_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)


mem = 36
run_time = "3-00:00:00"

pause_time = 30 #how much time (hours) to wait between jobs
pause_crit = 5 #how many jobs to do before pausing

#subj info
#stim info
model_dir = f'{curr_dir}/modelling'

stim_dir = f'/lab_data/behrmannlab/image_sets/'

#training info
model_arch = 'cornet_z_sl'

train_types = ['random', 'vggface_oneobject', 'imagenet_noface', 'imagenet_oneface', 'imagenet_vggface','vggface']


rand_seed = [2,3,4,5]
rand_seed= [1]
lr = .03
#lr = .003

def setup_sbatch(job_name, script_path):
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={job_name}

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1

# Job memory request
#SBATCH --mem={mem}gb

# Time limit days-hrs:min:sec
#SBATCH --time {run_time}

# Exclude
#SBATCH --exclude=mind-1-23

# Standard output and error log
#SBATCH --output={model_dir}/slurm_out/{job_name}.out

conda activate ml_new

{script_path}

"""
    return sbatch_setup



rois = ['FFA_face','EVC_face']
n = 0
for roi in rois:
    for age in ages:
        for train_type in train_types:
            
            job_name = f'optimal_ims_{age}_{train_type}'
            print(job_name)
            #os.remove(f"{job_name}.sh")
            
            f = open(f"{job_name}.sh", "a")
            
            script_path = f'python {model_dir}/find_optimal_image.py {train_type} {roi} {age}'
            f.writelines(setup_sbatch(job_name, script_path))
            
            
            f.close()
            
            subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
            os.remove(f"{job_name}.sh")
            n+=1

            if n >= pause_crit:
                #wait X hours
                time.sleep(pause_time*60)
                n = 0



""" 
for train_type in train_types:
    for rs in rand_seed:
        job_name = f'{model_arch}_{train_type}'
        print(job_name)
        #os.remove(f"{job_name}.sh")
        
        f = open(f"{job_name}.sh", "a")
        
        script_name= f'python supervised_training.py --data /lab_data/behrmannlab/image_sets/{train_type}/ --arch {model_arch} --rand_seed {rs}'
        f.writelines(setup_sbatch(job_name, script_path))
        
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")
        n+=1

        if n >= pause_crit:
            #wait X hours
            time.sleep(pause_time*60*60)
            n = 0






 """

