"""
Do RSA, but comparing the similarity between each timepoint
"""
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import warnings
import os, argparse
from matplotlib.pyplot import subplot

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb
import ginn_params as params
import analysis_funcs

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
vid = 'Aeronaut'
vols = params.vols

#training info
model_archs = ['cornet_z_sl']
'''
set model params
'''

train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']

train_dir = f'/lab_data/behrmannlab/image_sets/'
#n_classes = len(glob(f'{args.data}/train/*'))
#layer =['aIT','pIT'] #set in descending order
layer_type = ['aIT']




    

def create_rdm(ts):
    """
    Create RDM
    """
    
    rdm = np.corrcoef(ts)
    rdm_vec = rdm[np.triu_indices(n=ts.shape[0],k=1)] #remove lower triangle
    #fisher-z the rdm vec
    rdm_vec = np.arctanh(rdm_vec)
    return rdm, rdm_vec

"""
Calculate RDM for each model and layer
"""

for model_arch in model_archs:
    for train_type in train_types:
        for layer in layer_type:
            print(f'extracting rdms for {model_arch}_{train_type}_{layer}')
            model_ts = np.load(f'{data_dir}/{model_arch}_{train_type}_{layer}_{vid}_ts.npy')
            if use_pc_thresh == True:
                n_comps = analysis_funcs.calc_pc_n(analysis_funcs.extract_pc(model_ts), 0.9)
                
            pca = analysis_funcs.extract_pc(model_ts, n_comps)
            model_ts = pca.transform(model_ts)
            
            
            rdm, rdm_vec = create_rdm(model_ts)
            
            np.save(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_rdm_full.npy',rdm)
            np.save(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_rdm.npy',rdm_vec)
            

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