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

human_predict = False
model_predict = False
custom_predict = True
suf = '_1'
'''
model predictors
'''

use_pc_thresh = True
pc_perc = .99


study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)



'''
neural predictors
'''
ages = ['adult']
rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']
rois = ['LOC','FFA','A1','EVC']
file_suf = '_3'



exp = sys.argv[1]
analysis_type = sys.argv[2]
model_arch = sys.argv[3]
train_type = sys.argv[4]
layer = sys.argv[5]



if analysis_type == 'mean_movie_crossval':
    #predicts individual MVTS; PCA decomposition; reutrns mean correlation for each sub
    import mean_ts_movie_crossval as predict_script
    predict_ts = predict_script.predict_ts
elif analysis_type == 'mean_sub_crossval':
    #predicts mean TS; cross-vals by splitting subjects
    import mean_ts_sub_crossval as predict_script
    predict_ts = predict_script.predict_ts
    
suf = predict_script.suf



predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
summary_type = 'model'
print(exp, summary_type, analysis_type)
sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['architecture', 'train_type', 'layer'])

print(f'predicting using {model_arch} {train_type} {layer}...')
predictor_ts = np.load(f'{predictor_dir}/{model_arch}_{train_type}_{layer}_{vid}_ts.npy')

#standardize predictor_ts
predictor_ts = stats.zscore(predictor_ts, axis=0)

#convert nans to 0
predictor_ts[np.isnan(predictor_ts)] = 0


pca = analysis_funcs.extract_pc(predictor_ts)
predictor_comps = pca.transform(predictor_ts)
#standardize predictor_ts
predictor_comps = stats.zscore(predictor_comps, axis=0)


#predictor_ts = np.transpose(predictor_ts)
predictor_summary, boot_summary = predict_ts(predictor_comps, exp)


predictor_summary['architecture'] = model_arch
predictor_summary['train_type'] = train_type
predictor_summary['layer'] = layer

sub_summary = sub_summary.append(predictor_summary)
        
#save summary        
sub_summary.to_csv(f'{curr_dir}/results/mean_ts/seperated/{exp}_{model_arch}_{train_type}_{layer}_{analysis_type}{file_suf}.csv', index=False)

#save bootstrapped summary
boot_summary.to_csv(f'{curr_dir}/results/mean_ts/seperated/{exp}_{model_arch}_{train_type}_{layer}_{analysis_type}_boot{file_suf}.csv', index=False)