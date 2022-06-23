import os
from glob import glob as glob
import pdb

train_type = ['vggface', 'imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject']

out_dir = '/lab_data/behrmannlab/vlad/ginn/model_weights'
old_arch = 'cornet_z_sl'
new_arch = 'cornet_z_cl'

old_epoch = [2,6,11,16,21,26,31]
new_epoch = [1,5,10,15,20,25,30]


for tt in train_type:
    

    for ee in enumerate(old_epoch):
        file_type = f'{out_dir}/{old_arch}_{tt}_{ee[1]}_'
        files = glob(f'{file_type}*')
        
        
        try:
            new_file = files[0].replace(f'_{ee[1]}_',f'_{new_epoch[ee[0]]}_')
            os.rename(files[0], new_file)
        except:
            continue
    '''
    for ff in files:
        new_file = ff.replace(old_arch, old_arch.replace(old_arch,new_arch))
        
        os.rename(ff, new_file)
    ''' 