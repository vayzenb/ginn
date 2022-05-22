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

from nilearn import image, datasets
import nibabel as nib
'''
parser = argparse.ArgumentParser(description='Child MVPD')
parser.add_argument('--roi', required=False,
                    help='age of subjects to process', 
                    default='rLO')   
args = parser.parse_args()
'''
# threshold for PCA
pc_thresh = .9
clf = Ridge()

#set directories
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn/fmri'
exp_dir= f'ginn/fmri/hbn'
study_dir = f'/lab_data/behrmannlab/scratch/vlad/{exp_dir}'
subj_dir=f'{study_dir}/derivatives/preprocessed_data'
out_dir = f'{study_dir}/derivatives/mean_func'
roi_dir = f'{study_dir}/derivatives/rois'
curr_subs = pd.read_csv(f'{curr_dir}/HBN-Site-CBIC.csv')
curr_subs= curr_subs[curr_subs['Age']<8]

#load whole brain mask
whole_brain_mask = image.load_img('/opt/fsl/6.0.3/data/standard/MNI152_T1_2mm_brain.nii.gz')
affine = whole_brain_mask.affine
whole_brain_mask = image.binarize_img(whole_brain_mask)

seed_roi = 'rLO'

#Rois to predict
test_rois = ['LO', 'FFA', 'OFA']



def get_existing_files(curr_subs):
    
    sub_list =[]
    for sub in curr_subs:
        img = f'{subj_dir}/sub-{sub}/sub-{sub}_task-movieDM_bold.nii.gz'
        
        if os.path.exists(img):
            sub_list.append(sub)

    return sub_list




def extract_mv_ts(bold_vol, mask_dir):
    """
    extract multivariate time course from ROI
    """

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

    print('Seed data extracted...')

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

def calc_mvpd(seed_comps,target_comps, target_pca):
    """
    Conduct regression by iteratively fitting all seed PCs to target PCs

    
    """

    all_scores = []
    for pcn in range(0,len(target_pca.explained_variance_ratio_)):
        
        clf.fit(seed_comps, target_comps[:,pcn]) #fit seed PCs to target
        r_squared = clf.score(target_comps[:,pcn]) 
        pdb.set_trace()
        weighted_corr = r_squared * target_pca.explained_variance_ratio_[pcn]
        all_scores.append(weighted_corr)

    final_corr = np.sum(all_scores)/(np.sum(target_pca.explained_variance_ratio_))

    return final_corr


'''Extract adult data from ROI'''
adult_brain = image.get_data(image.load_img(f'{out_dir}/mean_task-movieDM_bold_18.nii.gz'))
seed_roi = f'{roi_dir}/{seed_roi}.nii.gz'
adult_ts = extract_mv_ts(adult_brain, seed_roi)

n_comp = calc_pc_n(extract_pc(adult_ts),pc_thresh) #determine number of PCs in train_data using threshold        
adult_pca = extract_pc(adult_ts, n_comp) #conduct PCA one more time with that number of PCs
adult_comps = adult_pca.transform(adult_ts) #transform train data in PCs

sub_list = get_existing_files(curr_subs['participant_id'])
for sub in sub_list:
    for roi in test_rois:
        for lr in ['l','r']:
            sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{lr}{roi}_ts_all.npy')

            n_comp = calc_pc_n(extract_pc(sub_ts),pc_thresh) #determine number of PCs in train_data using threshold        
            child_pca = extract_pc(sub_ts, n_comp) #conduct PCA one more time with that number of PCs
            child_comps = child_pca.transform(sub_ts) #transform train data in PCs

            score = calc_mvpd(adult_comps,child_comps, child_pca)
            pdb.set_trace()


