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

pause_time = 5 #how much time (minutes) to wait between jobs
pause_crit = 12 #how many jobs to do before pausing

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, fmri_suf, start_trs,end_trs, data_dir, vols, tr, fps, bin_size, ages= params.load_params(exp)
suf = ''

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


""" #Predict using model combos
model_types = ['imagenet_noface', 'vggface']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']
n= 0
for model1 in model_types:
    for layer1 in layers:
        for model2 in model_types:
            for layer2 in layers:

                if model1 == model2:
                    continue

                job_name = f'{exp}_predict_{model1}_{model2}_{layer1}_{layer2}{suf}'
                print(job_name)
                script_path = f'python {curr_dir}/exps/predict_combined_model.py {exp} mean_movie_crossval cornet_z_sl {model1} {model2} {layer1} {layer2}'

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


#Predict individual using model ts
model_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']
sub_layers = ['output', 'output', 'output', 'output', 'output', 'avgpool']

""" n = 0
#predict ts
for model in model_types:
    for layer in layers:
        job_name = f'{exp}_predict_ts_{model}_{layer}{suf}'
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
