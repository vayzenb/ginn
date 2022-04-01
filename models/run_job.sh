#!/bin/bash -l

# Job name
#SBATCH --job-name=classify_mixed
# Mail events (NONE, BEGIN, END, FAIL, ALL)
###############################################
########## example #SBATCH --mail-type=END,FAIL 
##############################################
#SBATCH --mail-type=ALL
#SBATCH --mail-user=vayzenb@cmu.edu
 
# Submit job to cpu queue                
#SBATCH -p gpu

#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-1,mind-1-3,mind-1-5,mind-1-26,mind-1-32 
# Job memory request
#SBATCH --mem=24gb

# Time limit days-hrs:min:sec
#SBATCH --time 3-00:00:00

# Standard output and error log
#SBATCH --output=slurm_out/mixed_classify.out

mkdir -p /scratch/vayzenbe/
rsync -a /lab_data/behrmannlab/image_sets/mixed_imagenet_vggface /scratch/vayzenbe/

python cornet_classify_training.py

rm -rf /scratch/vayzenbe/mixed_imagenet_vggface