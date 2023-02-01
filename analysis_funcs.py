import os
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA


def get_existing_files(curr_subs, subj_dir, file_suf):
    
    sub_file =pd.DataFrame(columns=['sub','age'])
    for sub in enumerate(curr_subs['participant_id']):
        img = f'{subj_dir}/{sub[1]}/{sub[1]}_task-{file_suf}_bold.nii.gz'
        
        if os.path.exists(img):
            
            sub_file = sub_file.append(pd.Series([sub[1], curr_subs['Age'][sub[0]]], index = sub_file.columns), ignore_index = True)

    return sub_file

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