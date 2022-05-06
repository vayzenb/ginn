#!/bin/bash -l


# Job name
#SBATCH --job-name=inversion_pref

#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu

# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Job memory request
#SBATCH --mem=24gb

# Time limit days-hrs:min:sec
#SBATCH --time 3-00:00:00

# Exclude
#SBATCH --exclude=mind-1-26,mind-1-30

# Standard output and error log
#SBATCH --output=inversion_pref.out

conda activate ml_new

# python exps/identification_recog_svm.py
python exps/inversion_pref.py

