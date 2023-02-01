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

import ginn_params as params
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import metrics

from nilearn import image, signal
import nibabel as nib
print('Libraries loaded...')
# threshold for PCA

use_pc_thresh = True
n_comp = 10
global_signal = 'mean'

pc_thresh = .9

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

exp = params.exp
exp_dir = params.exp_dir
file_suf = params.file_suf
fix_tr = params.fix_tr

data_dir = params.data_dir
study_dir = params.study_dir

sub_list = params.sub_list

file_suf = params.file_suf



out_dir = data_dir


roi_dir = f'{study_dir}/derivatives/rois'

rois = ['LO','FFA','A1']


#curr_subs= curr_subs[curr_subs['Age']<8]

#load whole brain mask
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)



def extract_pc(data, n_components=None):

    """
    Extract principal components
    if n_components isn't set, it will extract all it can
    """
    
    pca = PCA(n_components = n_components)
    pca.fit(data)
    
    return pca

# %%
def calc_pc_n(pca, thresh):
    '''
    Calculate how many PCs are needed to explain X% of data
    
    pca - result of pca analysis
    thresh- threshold for how many components to keep
    '''
    
    explained_variance = pca.explained_variance_ratio_
    
    var = 0
    for n_comp, ev in enumerate(explained_variance):
        var += ev #add each PC's variance to current variance
        #print(n_comp, ev, var)

        if var >=thresh: #once variance > than thresh, stop
            break
    return n_comp+1

def create_rdm(ts):
    """
    Create RDM
    """
    
    #rdm = 1-metrics.pairwise.cosine_similarity(ts)
    #rdm = euclidean_distances(ts,squared = True)
    rdm = np.corrcoef(ts)
    rdm_vec = rdm[np.triu_indices(n=ts.shape[0],k=1)] #remove lower triangle
    
    #fisher-z the rdm vec
    rdm_vec = np.arctanh(rdm_vec)
    
    return rdm, rdm_vec

def create_mean_rdm(age):


    """create mean RDM for particular age"""
    print(f'Creating mean RDM for {age}...')
    if age == 'infant':
        curr_subs = sub_list[sub_list['Age']<18]
    elif age == 'adult':
        curr_subs = sub_list[sub_list['Age']>=18]
    
    for roi in rois:
        for lr in ['l','r']:
            all_rdms = []
            for sub in enumerate(curr_subs['participant_id']):
                #load rdm for each subject
                rdm = pd.read_csv(f'{out_dir}/sub-{sub[1]}/rdms/{lr}{roi}_rdm_vec.csv')
                all_rdms.append(rdm['rdm'])
            
            #average rdms
            mean_rdm = np.mean(np.stack(all_rdms),axis = 0)
            
            np.save(f'{out_dir}/group_func/{lr}{roi}_{age}_rdm.npy',mean_rdm)

        

def extract_rdm():
    """
    Calculate RDM for each ROI
    """
    for roi in rois:
        for lr in ['l','r']:

            
            #print(f'predicting for: {slr}{sr}', seed_comps.shape[1])
            for sub in enumerate(sub_list['participant_id']):
                print(f'Extracting RDMs for sub: {sub[1]}', f'{sub[0]} out of {len(sub_list)}')
                whole_ts = np.load(f'{data_dir}/sub-{sub[1]}/timeseries/whole_brain_ts.npy')
                whole_ts = whole_ts[fix_tr:,:]
                #print(f'predicting {sub} from {args.roi}', seed_comps.shape[1], f'{sub[0]+1} of {len(sub_list)}')
                os.makedirs(f'{out_dir}/sub-{sub[1]}/rdms', exist_ok=True)

                #load data
                sub_ts = np.load(f'{data_dir}/sub-{sub[1]}/timeseries/{lr}{roi}_ts_all.npy')
                sub_ts = sub_ts[fix_tr:,:] #remove fix TRs
                
                #remove global signal
                if global_signal == 'pca':
                    pca = extract_pc(whole_ts, n_components = 10)
                    whole_confound = pca.transform(whole_ts)
                elif global_signal == 'mean':
                    whole_confound = np.mean(whole_ts,axis =1)

                sub_ts = signal.clean(sub_ts,confounds = whole_confound, standardize_confounds=True)
                
                
                #if use_pc_thresh == True: n_comp = calc_pc_n(extract_pc(sub_ts),pc_thresh) #determine number of PCs in train_data using threshold        

                #sub_pcwhole_a = extract_pc(sub_ts, n_comp) #conduct PCA one more time with that number of PCs
                #sub_pcs = sub_pca.transform(sub_ts) #transform train data in PCs
                
                rdm, rdm_vec = create_rdm(sub_ts)
                rdm_df = pd.DataFrame(rdm_vec, columns = ['rdm'])
                

                np.save(f'{out_dir}/sub-{sub[1]}/rdms/{lr}{roi}_rdm.npy',rdm)
                rdm_df.to_csv(f'{out_dir}/sub-{sub[1]}/rdms/{lr}{roi}_rdm_vec.csv',index = False)



    # %%

#extract_rdm()
create_mean_rdm('infant')
create_mean_rdm('adult')