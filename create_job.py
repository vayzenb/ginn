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

mem = 36
run_time = "3-00:00:00"



def setup_sbatch(job_name, script_name):
    sbatch_setup = f"""#!/bin/bash -l
# Job name
#SBATCH --job-name={job_name}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
# Submit job to cpu queue                
#SBATCH -p gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
# Job memory request
#SBATCH --mem={mem}gb
# Time limit days-hrs:min:sec
#SBATCH --time {run_time}
# Exclude
# SBATCH --exclude=mind-1-26,mind-1-30
# Standard output and error log
#SBATCH --output={curr_dir}/slurm_out/{job_name}.out
conda activate ml_new

{script_name}
"""
    return sbatch_setup


'''
# run low-demand scripts
for script in script_list:
    job_name = script
    script_path = f'python {curr_dir}/eeg/{script}.py'
    print(job_name)
    #create sbatch script
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(job_name, script_path))
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")
'''


model_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
model_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']
layers = ['decoder']
sub_layers = ['output', 'output', 'output', 'output', 'output', 'avgpool']
sub_layers = ['avgpool']
#run high-demand scripts
for model in model_types:
    for layer in layers:
        job_name = f'{model}_{layer}'
        script_path = f'python {curr_dir}/modelling/extract_model_ts.py {model} {layer} {sub_layers[layers.index(layer)]}'
        print(job_name)
        #create sbatch script
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(job_name, script_path))
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")

'''
for sub in sub_list:
    job_name = f'tga_{sub}'
    script_path = f'python analysis/time_generalization.py {sub}'
    print(job_name)

    #create sbatch script
    f = open(f"{job_name}.sh", "a")
    f.writelines(setup_sbatch(job_name, script_path))
    
    f.close()
    
    subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
    os.remove(f"{job_name}.sh")
'''