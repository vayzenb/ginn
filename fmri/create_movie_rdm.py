"""
Do RSA, but comparing the similarity between each timepoint
"""

import warnings
import os, argparse
from matplotlib.pyplot import subplot
from sqlalchemy import column, false
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import pairwise, pairwise_distances

from nilearn import image, datasets
import nibabel as nib
print('Libraries loaded...')
# threshold for PCA

use_pc_thresh = True
n_comp = 10

pc_thresh = .9

clf = LinearRegression()

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
exp_dir= f'ginn/fmri/hbn'
study_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{study_dir}/derivatives/preprocessed_data'
out_dir = f'/lab_data/behrmannlab/vlad/{exp_dir}/'
results_dir =f'{curr_dir}/results/rsa'
roi_dir = f'{study_dir}/derivatives/rois'
curr_subs = pd.read_csv(f'{curr_dir}/fmri/HBN-Site-CBIC.csv')
rois = ['LO','FFA','OFA']



#curr_subs= curr_subs[curr_subs['Age']<8]

#load whole brain mask
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)


def get_existing_files(curr_subs):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/sub-{sub[1]}/sub-{sub[1]}_task-movieDM_bold.nii.gz'
        
        if os.path.exists(img):
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file



def extract_mv_ts(bold_vol, mask_dir):
    """
    extract multivariate time course from ROI
    """
    print('Extracting MV time series...')
    #load seed
    roi = image.get_data(image.load_img(f'{mask_dir}'))
    #Just ensure its all binary
    roi[roi>0] = 1
    roi[roi<=0] = 0
    reshaped_roi = np.reshape(roi, whole_brain_mask.shape +(1,))
    masked_img = reshaped_roi*bold_vol

    #extract voxel resposnes from within mask
    mv_ts = masked_img.reshape(-1, bold_vol.shape[3]) #reshape into rows (voxels) x columns (time)
    mv_ts =mv_ts[~np.all(mv_ts == 0, axis=1)] #remove voxels that are 0 (masked out)
    mv_ts = np.transpose(mv_ts)

    

    return mv_ts


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

def create_rdm(sub_pcs):
    """
    Create RDM
    """
    
    
    rdm = pairwise.cosine_similarity(sub_pcs) #extract cosine similarity matrix
    rdm = rdm[np.triu_indices(n=750,k=1)] #remove lower triangle
    
    return rdm

"""
Calculate RDM for each ROI
"""

sub_list = get_existing_files(curr_subs)
sub_list = sub_list.drop_duplicates()
sub_list = sub_list.reset_index()


sub_summary = pd.DataFrame(columns = ['sub','age','roi', 'r2'])

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
