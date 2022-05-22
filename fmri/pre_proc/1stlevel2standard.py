
import os, argparse, shutil
import subprocess

parser = argparse.ArgumentParser(description='HBN preprocessing')
parser.add_argument('--path', required=True,
                    help='path to data directory', 
                    default=None)   
parser.add_argument('--anat', required=True,
                    help='what anatomical file to register to', 
                    default=None)
parser.add_argument('--sub', required=True,
                    help='path to subject folder', 
                    default=None)      

suf = ''

args = parser.parse_args()

out_dir = f'{args.path}/derivatives/preprocessed_data/{args.sub}'
#Make relevant dirs
os.makedirs(out_dir, exist_ok=True)

sub_file = f'{args.path}/{args.sub}/derivatives/fsl/1stLevel.feat'
raw_func = f'{sub_file}/filtered_func_data.nii.gz'
stand_func = f'{out_dir}/{args.sub}_task-movieDM_bold.nii.gz'

bash_cmd = f"flirt -in {raw_func} -ref {args.anat} -out {stand_func} -applyxfm -init {sub_file}/reg/example_func2standard.mat -interp trilinear"
print(bash_cmd)
subprocess.run(bash_cmd.split(), check=True)