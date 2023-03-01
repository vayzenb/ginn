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
import analysis_setup

#analysis scripts
exp = 'aeronaut'

human_predict = False
model_predict = True
suf = ''
'''
model predictors
'''
model_archs = ['cornet_z_sl']
train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']




use_pc_thresh = True
pc_perc = .99


study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)



'''
neural predictors
'''
ages = ['adult']
rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']

file_suf = ''

group_type = 'mean'


if len(sys.argv) > 1:
    analysis_type = sys.argv[1]
else:
    analysis_type = 'mean_movie_crossval'


if analysis_type == 'mean_movie_crossval':
    #predicts individual MVTS; PCA decomposition; reutrns mean correlation for each sub
    import mean_ts_movie_crossval as predict_script
    predict_ts = predict_script.predict_ts
elif analysis_type == 'mean_sub_crossval':
    #predicts mean TS; cross-vals by splitting subjects
    import mean_ts_sub_crossval as predict_script
    predict_ts = predict_script.predict_ts
    
suf = predict_script.suf

print(analysis_type)


predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
summary_type = 'model'
sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['architecture', 'train_type', 'layer'])

for model_arch in model_archs:
    for train_type in train_types:
        for layer in layers:
            print(f'predicting using {model_arch} {train_type} {layer}...')

            predictor_summary =analysis_setup.setup_predictors(exp, analysis_type, model_arch, train_type, layer)

            sub_summary = sub_summary.append(predictor_summary)

    sub_summary.to_csv(f'{curr_dir}/results/mean_ts/{exp}_{model_arch}_{analysis_type}{file_suf}.csv', index=False)
