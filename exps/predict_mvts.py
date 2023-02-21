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
model_predict = True
suf = ''
'''
model predictors
'''
model_arch = ['cornet_z_sl']
train_type = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer = ['V1','V2','V4','pIT','aIT', 'decoder']



use_pc_thresh = True
pc_perc = .99


study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)



'''
neural predictors
'''
ages = ['adult']
rois = ['LOC','FFA','A1','EVC'] + ['lLOC','lFFA','lA1','lEVC'] + ['rLOC','rFFA','rA1','rEVC']
#rois = ['LOC','FFA','A1','EVC']
file_suf = '_og'

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
if human_predict == True:

    predictor_dir = f'/lab_data/behrmannlab/vlad/ginn/fmri/{exp}/derivatives/group_func'
    summary_type = 'human'


    sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['seed_age', 'seed_roi'])
    for age in ages:
        for rr in rois:
            
            roi = f'{rr}'
            print(f'predicting using {age} and {roi}...')
            predictor_ts = np.load(f'{predictor_dir}/{group_type}_{roi}_{age}_ts.npy')
            #standardize predictor_ts
            
            

            if group_type == 'mean':
                if use_pc_thresh == True:
                    n_comps = analysis_funcs.calc_pc_n(analysis_funcs.extract_pc(predictor_ts), pc_perc)
                
                pca = analysis_funcs.extract_pc(predictor_ts, n_comps)
                predictor_ts = pca.transform(predictor_ts)


                predictor_ts = predictor_ts[fix_tr:,:]
                #predictor_ts = np.transpose(predictor_ts)
                predictor_summary = predict_ts(predictor_ts)
                
                predictor_summary['seed_age'] = age
                predictor_summary['seed_roi'] = roi

                sub_summary = sub_summary.append(predictor_summary)
                
                
        
    sub_summary.to_csv(f'{curr_dir}/results/mean_ts/{exp}_{summary_type}_{analysis_type}{file_suf}.csv', index=False)

if model_predict == True:
    predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
    summary_type = 'model'
    sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['architecture', 'train_type', 'layer'])
    
    for mt in model_arch:
        for tt in train_type:
            for ll in layer:
                print(f'predicting using {mt} {tt} {ll}...')
                predictor_ts = np.load(f'{predictor_dir}/{mt}_{tt}_{ll}_{vid}_ts.npy')
                
                #standardize predictor_ts
                predictor_ts = stats.zscore(predictor_ts, axis=0)

                #convert nans to 0
                predictor_ts[np.isnan(predictor_ts)] = 0


                pca = analysis_funcs.extract_pc(predictor_ts)
                predictor_comps = pca.transform(predictor_ts)
                #standardize predictor_ts
                predictor_comps = stats.zscore(predictor_comps, axis=0)

                
                
                #predictor_ts = np.transpose(predictor_ts)
                predictor_summary,boot_summary = predict_ts(predictor_comps, exp)
                
                
                predictor_summary['architecture'] = mt
                predictor_summary['train_type'] = tt
                predictor_summary['layer'] = ll

                sub_summary = sub_summary.append(predictor_summary)

                #save bootstrapped summary
                boot_summary.to_csv(f'{curr_dir}/results/mean_ts/seperated/{exp}_{model_arch}_{train_type}_{layer}_{analysis_type}_boot{file_suf}.csv', index=False)
                

        sub_summary.to_csv(f'{curr_dir}/results/mean_ts/{exp}_{model_arch}_{analysis_type}{file_suf}.csv', index=False)
