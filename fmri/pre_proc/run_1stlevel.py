import os, argparse, shutil
import subprocess

parser = argparse.ArgumentParser(description='HBN preprocessing')
parser.add_argument('--path', required=True,
                    help='path to data directory', 
                    default=None)   
parser.add_argument('--og_sub', required=True,
                    help='original sub to copy from', 
                    default=None)
parser.add_argument('--curr_sub', required=True,
                    help='path to subject folder', 
                    default=None)      

suf = ''

args = parser.parse_args()

og_file = f'{args.path}/{args.og_sub}/derivatives/fsl/1stLevel{suf}.fsf'
new_file = f'{args.path}/{args.curr_sub}/derivatives/fsl/1stLevel{suf}.fsf'



#create directory
os.makedirs(f'{args.path}/{args.curr_sub}/derivatives/fsl/', exist_ok=True)

#copy feat file
shutil.copyfile(og_file, new_file)

# Read in the file
with open(new_file, 'r') as file :
  filedata = file.read()

# Replace the target string
filedata = filedata.replace(args.og_sub, args.curr_sub)

# Write the file out again
with open(new_file, 'w') as file:
  file.write(filedata)

#start feat
bash_cmd = f'feat {new_file}'
subprocess.run(bash_cmd.split())