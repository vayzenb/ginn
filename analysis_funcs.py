import os
import numpy as np
import pandas as pd
from nilearn import signal

from sklearn.decomposition import PCA
import pdb
print('libraries loaded')


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

def extract_roi_data(subj_dir, curr_subs, roi,roi_suf, fix_tr,global_signal):
    '''
    load subs into numpy array
    '''
    print(f'extracting {roi} data...')
    
   
    all_data = []
    for sub in curr_subs['participant_id']:
        whole_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/whole_brain_ts.npy')
        whole_ts = whole_ts[fix_tr:,:]

        sub_ts = np.load(f'{subj_dir}/sub-{sub}/timeseries/{roi}{roi_suf}.npy')
        sub_ts = sub_ts[fix_tr:,:]

        #pdb.set_trace()

        if global_signal != '':
            #remove global signal
            if global_signal == 'pca':
                pca = extract_pc(whole_ts, n_components = 10)
                whole_confound = pca.transform(whole_ts)
            elif global_signal == 'mean':
                whole_confound = np.mean(whole_ts,axis =1)
            
            

            sub_ts = signal.clean(sub_ts,confounds = whole_confound, standardize_confounds=True)
        
        


        

        sub_ts = np.transpose(sub_ts)
        #sub_ts = np.expand_dims(sub_ts,axis =2)
        
        sub_ts = np.mean(sub_ts, axis = 0)
        
        all_data.append(sub_ts)
        #sub_ts = np.reshape(sub_ts, [sub_ts.shape[1], sub_ts.shape[0], sub_ts.shape[2]])
        #pdb.set_trace()

    return all_data