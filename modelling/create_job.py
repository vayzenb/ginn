"""
iteratively creates sbatch scripts to run multiple jobs at once the
"""

import subprocess
from glob import glob
import os
import time
import pdb

mem = 36
run_time = "3-00:00:00"

pause_time = 16 #how much time (hours) to wait between jobs
pause_crit = 5 #how many jobs to do before pausing

#subj info
#stim info
study_dir = f'/user_data/vayzenbe/GitHub_Repos/ginn/modelling'

stim_dir = f'/lab_data/behrmannlab/image_sets/'

#training info
model_arch = 'cornet_z_sl'

train_type = ['vggface', 'vggface_oneobject', 'imagenet_noface', 'imagenet_oneface', 'imagenet_vggface']
#train_type = ['vggface']


rand_seed = [2,3,4,5]
#rand_seed= [2]
lr = .03
#lr = .003

def setup_sbatch(model, train_cat, seed):
    sbatch_setup = f"""#!/bin/bash -l


# Job name
#SBATCH --job-name={model}_{train_cat}

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
# SBATCH --exclude=mind-1-26,mind-1-30

# Standard output and error log
#SBATCH --output={study_dir}/slurm_out/{model}_{train_cat}_{seed}.out

conda activate ml_new

# python train_model.py /scratch/vayzenbe/{train_cat}/ --arch cornet_z --epochs 50 --nce-k 4096 --nce-t 0.07 --lr {lr} --nce-m 0.5 --low-dim 128 -b 256 --rand_seed {seed}
python supervised_training.py --data /lab_data/behrmannlab/image_sets/{train_cat}/ --arch {model_arch} --rand_seed {seed}
# python supervised_training.py --data /lab_data/behrmannlab/image_sets/{train_cat}/ --arch {model_arch} --resume /lab_data/behrmannlab/vlad/ginn/model_weights/{model_arch}_{train_cat}_15_{seed}.pth.tar
"""
    return sbatch_setup




n = 0
for tt in train_type:
    for rs in rand_seed:
        job_name = f'{model_arch}_{tt}'
        print(job_name)
        #os.remove(f"{job_name}.sh")
        
        f = open(f"{job_name}.sh", "a")
        f.writelines(setup_sbatch(model_arch, tt, rs))
        
        
        f.close()
        
        subprocess.run(['sbatch', f"{job_name}.sh"],check=True, capture_output=True, text=True)
        os.remove(f"{job_name}.sh")
        n+=1

        if n >= pause_crit:
            #wait X hours
            time.sleep(pause_time*60*60)
            n = 0








