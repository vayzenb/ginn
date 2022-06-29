"""
Do RSA, but comparing the similarity between each timepoint
"""

import warnings
import os, argparse
from matplotlib.pyplot import subplot

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import pairwise, pairwise_distances

print('Libraries loaded...')
# threshold for PCA

use_pc_thresh = True
n_comp = 10

pc_thresh = .9

clf = LinearRegression()

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
data_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/model_ts"
out_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/rdms"
results_dir =f'{curr_dir}/results/rsa'
vid = 'DM-clip'

#training info
model_archs = ['cornet_z_sl','cornet_z_cl']
'''
set model params
'''

train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']

train_dir = f'/lab_data/behrmannlab/image_sets/'
#n_classes = len(glob(f'{args.data}/train/*'))
#layer =['aIT','pIT'] #set in descending order
layer_type = ['aIT','pIT']




    

def create_rdm(ts):
    """
    Create RDM
    """
    
    rdm = np.corrcoef(ts) * -1
    rdm_vec = rdm[np.triu_indices(n=750,k=1)] #remove lower triangle
    
    return rdm, rdm_vec

"""
Calculate RDM for each model and layer
"""

for model_arch in model_archs:
    for train_type in train_types:
        for layer in layer_type:
            print(f'extracting rdms for {model_arch}_{train_type}_{layer}')
            model_ts = np.load(f'{data_dir}/{model_arch}_{train_type}_{layer}_{vid}_ts.npy')
            rdm, rdm_vec = create_rdm(model_ts)

            np.save(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_rdm.npy',rdm)
            rdm_df = pd.DataFrame(rdm_vec, columns = ['rdm'])
            rdm_df.to_csv(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_rdm_vec.csv',index = False)



'''
#print(f'predicting for: {slr}{sr}', seed_comps.shape[1])
for sub in enumerate(sub_list['sub']):
    print(f'Extracting RDMs for sub: {sub[1]}', f'{sub[0]} out of {len(sub_list)}')
    #print(f'predicting {sub} from {args.roi}', seed_comps.shape[1], f'{sub[0]+1} of {len(sub_list)}')
    os.makedirs(f'{out_dir}/sub-{sub[1]}/derivatives/rdms', exist_ok=True)
    for roi in rois:
        for lr in ['l','r']:
            
            sub_ts = np.load(f'{subj_dir}/sub-{sub[1]}/timeseries/{lr}{roi}_ts_all.npy')
            
            

            if use_pc_thresh == True: n_comp = calc_pc_n(extract_pc(sub_ts),pc_thresh) #determine number of PCs in train_data using threshold        

            sub_pca = extract_pc(sub_ts, n_comp) #conduct PCA one more time with that number of PCs
            sub_pcs = sub_pca.transform(sub_ts) #transform train data in PCs
            rdm = create_rdm(sub_pcs)
            rdm_df = pd.DataFrame(rdm, columns = ['rdm'])
            rdm_df.to_csv(f'{out_dir}/sub-{sub[1]}/derivatives/rdms/{lr}{roi}_rdm.csv',index = False)

            
# %%
'''