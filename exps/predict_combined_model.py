"""
Predict using model combos
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

file_suf = ''
'''
model predictors
'''



predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
summary_type = 'model'


def setup_predictors(exp, analysis_type, model_arch, train_type1,train_type2, layer1,layer2):
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
    sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['architecture', 'train_type1','train_type2', 'layer1', 'layer2'])

    print(f'predicting using {model_arch} {train_type1} {train_type2} {layer1} {layer2}...')
    predictor_ts = np.load(f'{predictor_dir}/{model_arch}_{"_".join([train_type1,train_type2])}_{"_".join([layer1,layer2])}_{vid}_ts.npy')


    predictor_ts = stats.zscore(predictor_ts, axis=0)

    #predictor_ts = np.transpose(predictor_ts)
    predictor_summary,boot_summary = predict_ts(predictor_ts, exp)


    predictor_summary['architecture'] = model_arch
    predictor_summary['train_type1'] = train_type1
    predictor_summary['train_type2'] = train_type2

    predictor_summary['layer1'] = layer1
    predictor_summary['layer2'] = layer2

    sub_summary = sub_summary.append(predictor_summary)

    #save bootstrapped summary
    boot_summary.to_csv(f'{curr_dir}/results/mean_ts/resamples/{exp}_{model_arch}_{"_".join([train_type1,train_type2])}_{"_".join([layer1,layer2])}_boot{file_suf}.csv', index=False)
    #save seperate summary for each sub
    predictor_summary.to_csv(f'{curr_dir}/results/mean_ts/seperated/{exp}_{model_arch}_{"_".join([train_type1,train_type2])}_{"_".join([layer1,layer2])}{file_suf}.csv', index=False)

    return predictor_summary


if len(sys.argv) > 1:

    exp = sys.argv[1]
    analysis_type = sys.argv[2]
    model_arch = sys.argv[3]
    train_type1 = sys.argv[4]
    train_type2 = sys.argv[5]
    layer1 = sys.argv[6]
    layer2 = sys.argv[7]

    setup_predictors(exp, analysis_type, model_arch, train_type1,train_type2, layer1,layer2)