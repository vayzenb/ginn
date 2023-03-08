"""
A generic script for predictiving multivariate time series (MVTs) using a variety of predictors.

"""

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}/fmri')
sys.path.insert(1, f'{curr_dir}')

from scipy import stats
import pandas as pd
import numpy as np
import pdb

import ginn_params as params
import analysis_funcs

#analysis scripts
exp = 'aeronaut'

suf = ''
'''
model predictors
'''

use_pc_thresh = True
pc_perc = .99
global n_comps
n_comps = 0

#check if n_comps is set
if n_comps == 0:
    n_comps = 90

exp = 'aeronaut'
analysis_type = 'mean_movie_crossval'
model_arch = 'cornet_z_sl'
train_type = 'vggface'
layer = 'pIT'


'''
neural predictors
'''

file_suf = ''



predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
summary_type = 'model'

def setup_predictors(exp, analysis_type, model_arch, train_type, layer):
    #load exp params
    study_dir,subj_dir, sub_list, vid, fmri_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

    #Select analysis script
    if analysis_type == 'mean_movie_crossval':
        #predicts individual MVTS; PCA decomposition; reutrns mean correlation for each sub
        import mean_ts_movie_crossval as predict_script
        predict_ts = predict_script.predict_ts
    elif analysis_type == 'mean_sub_crossval':
        #predicts mean TS; cross-vals by splitting subjects
        import mean_ts_sub_crossval as predict_script
        predict_ts = predict_script.predict_ts
        
    suf = predict_script.suf

    
    print(exp, summary_type, analysis_type)
    sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['architecture', 'train_type', 'layer'])

    print(f'predicting using {model_arch} {train_type} {layer}...')
    predictor_ts = np.load(f'{predictor_dir}/{model_arch}_{train_type}_{layer}_{vid}_ts.npy')

    #standardize predictor_ts
    predictor_ts = stats.zscore(predictor_ts, axis=0)

    #convert nans to 0
    predictor_ts[np.isnan(predictor_ts)] = 0
    print(n_comps)
    
    


    pca = analysis_funcs.extract_pc(predictor_ts,n_components=n_comps)
    predictor_comps = pca.transform(predictor_ts)
    #standardize predictor_ts
    predictor_comps = stats.zscore(predictor_comps, axis=0)

    #predictor_ts = np.transpose(predictor_ts)
    predictor_summary,boot_summary = predict_ts(predictor_comps, exp)


    predictor_summary['architecture'] = model_arch
    predictor_summary['train_type'] = train_type
    predictor_summary['layer'] = layer

    sub_summary = sub_summary.append(predictor_summary)

    #save bootstrapped summary
    boot_summary.to_csv(f'{curr_dir}/results/mean_ts/resamples/{exp}_{model_arch}_{train_type}_{layer}_{analysis_type}_boot{file_suf}.csv', index=False)
    #save seperate summary for each sub
    predictor_summary.to_csv(f'{curr_dir}/results/mean_ts/seperated/{exp}_{model_arch}_{train_type}_{layer}_{analysis_type}{file_suf}.csv', index=False)

    return predictor_summary


if len(sys.argv) > 1:

    exp = sys.argv[1]
    analysis_type = sys.argv[2]
    model_arch = sys.argv[3]
    train_type = sys.argv[4]
    layer = sys.argv[5]


setup_predictors(exp, analysis_type, model_arch, train_type, layer)




