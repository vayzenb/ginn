"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""
curr_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn' #CHANGE AS NEEEDED CAUSE ITS FOR VLAAAD

import sys
sys.path.insert(0,curr_dir)

import subprocess
from glob import glob
import os
import time
import pdb
import ginn_params as params

mem = 24
run_time = "3-00:00:00"

pause_time = 2 #how much time (minutes) to wait between jobs
pause_crit = 15 #how many jobs to do before pausing

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
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
# SBATCH --exclude=mind-1-26,mind-1-30
# Standard output and error log
#SBATCH --output={curr_dir}/slurm_out/{job_name}.out

conda activate fmri_new

{script_name}
"""
    return sbatch_setup




model_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
model_types = ['imagenet_noface']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']
model_types = ['imagenet_noface']
layers = ['V1','V2']

sub_layers = ['output', 'output', 'output', 'output', 'output', 'avgpool']

n = 0
#predict ts
for model in model_types:
    for layer in layers:
        job_name = f'predict_ts_{model}_{layer}'
        script_path = f'python {curr_dir}/exps/analysis_setup.py {exp} mean_movie_crossval cornet_z_sl {model} {layer}'
        print(job_name)
        #create sbatch script
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(job_name, script_path))
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh") 
    
        n+=1
        if n >= pause_crit:
            #wait X minutes
            time.sleep(pause_time*60)
            n = 0


""" 
#extract model ts
n = 0
for model in model_types:
    for layer in layers:
        job_name = f'extract_ts_{model}_{layer}'
        script_path = f'python {curr_dir}/modelling/extract_model_ts.py {model} {layer} {sub_layers[layers.index(layer)]}'
        print(job_name)
        #create sbatch script
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(job_name, script_path))
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh") 
    
        n+=1
        if n >= pause_crit:
            #wait X minutes
            time.sleep(pause_time*60)
            n = 0 """
'''
#extract fmri ts
n =0 
for sub in sub_list['participant_id']:
    job_name = f'extract_ts_{sub}'
    script_path = f'python fmri/extract_fmri_timeseries.py {sub}'
    print(job_name)

    #create sbatch script
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(job_name, script_path))
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")

    n+=1
    if n >= pause_crit:
        #wait X minutes
        time.sleep(pause_time*60)
        n = 0
'''