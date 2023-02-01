"""
A generic script for predictiving multivariate time series (MVTs) using a variety of predictors.

"""

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}/fmri')
sys.path.insert(1, f'{curr_dir}')


import pandas as pd
import numpy as np
import pdb

import ginn_params as params
import analysis_funcs

#analysis scripts


human_predict = True
model_predict = True
suf = ''
'''
model predictors
'''
model_arch = ['cornet_z_sl']
train_type = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer = ['aIT']

use_pc_thresh = True
n_comps = 10

exp = params.exp
vid = params.vid
fix_tr = params.fix_tr

'''
neural predictors
'''
ages = ['adult']
rois = ['LO','FFA']
group_type = 'mean'

#numober of SRM feats to use as predictors
n_feats = [25,50,100,200]
n_feats = [25]

if len(sys.argv) > 1:
    analysis_type = sys.argv[1]
else:
    analysis_type = 'indiv'

print(analysis_type)

if analysis_type == 'indiv':
    #predicts individual MVTS; PCA decomposition; reutrns mean correlation for each sub
    import mvpd_indiv as predict_script
    predict_ts = predict_script.predict_indvidual
elif analysis_type == 'mv_movie_crossval':
    #predicts group MVTS; uses SRM decomposition; cross-vals by splitting movie
    import mvpd_movie_crossval as predict_script
    predict_ts = predict_script.predict_srm
elif analysis_type == 'mv_sub_crossval':
    #predicts group MVTS; uses SRM decomposition; cross-vals by splitting subjects
    import mvpd_sub_crossval as predict_script
    predict_ts = predict_script.predict_srm
elif analysis_type == 'mean_sub_crossval':
    #predicts mean TS; cross-vals by splitting subjects
    import mean_ts_sub_crossval as predict_script
    predict_ts = predict_script.predict_ts
    
suf = predict_script.suf

print(n_comps)
if human_predict == True:

    predictor_dir = f'/lab_data/behrmannlab/vlad/ginn/fmri/{exp}/derivatives/group_func'
    summary_type = 'human'

    for n_feat in n_feats:
        sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['seed_age', 'seed_roi'])
        for age in ages:
            for rr in rois:
                for lr in ['l','r']:
                    roi = f'{lr}{rr}'
                    print(f'predicting using {age} and {roi}...')
                    predictor_ts = np.load(f'{predictor_dir}/{group_type}_{roi}_{age}_ts.npy')
                    

                    if group_type == 'mean':
                        if use_pc_thresh == True:
                            n_comps = analysis_funcs.calc_pc_n(analysis_funcs.extract_pc(predictor_ts), 0.9)
                        
                        pca = analysis_funcs.extract_pc(predictor_ts, n_comps)
                        predictor_ts = pca.transform(predictor_ts)


                    predictor_ts = predictor_ts[fix_tr:,:]
                    #predictor_ts = np.transpose(predictor_ts)
                    predictor_summary = predict_ts(predictor_ts)
                    
                    predictor_summary['seed_age'] = age
                    predictor_summary['seed_roi'] = roi

                    sub_summary = sub_summary.append(predictor_summary)
                    
                    
        
        sub_summary.to_csv(f'{curr_dir}/results/mvpd/{exp}_{summary_type}_{analysis_type}.csv', index=False)

if model_predict == True:
    predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
    summary_type = 'model'
    sub_summary = pd.DataFrame(columns=predict_script.summary_cols + ['architecture', 'train_type', 'layer'])
    
    for mt in model_arch:
        for tt in train_type:
            for ll in layer:
                print(f'predicting using {mt} {tt} {ll}...')
                predictor_ts = np.load(f'{predictor_dir}/{mt}_{tt}_{ll}_{vid}_ts.npy')
                #predictor_ts = predictor_ts[fix_tr:,:]
                #pdb.set_trace()

                if use_pc_thresh == True:
                    n_comps = analysis_funcs.calc_pc_n(analysis_funcs.extract_pc(predictor_ts), 0.9)
                
                pca = analysis_funcs.extract_pc(predictor_ts, n_comps)
                predictor_ts = pca.transform(predictor_ts)
                
                #predictor_ts = np.transpose(predictor_ts)
                predictor_summary = predict_ts(predictor_ts)
                
                
                predictor_summary['architecture'] = mt
                predictor_summary['train_type'] = tt
                predictor_summary['layer'] = ll

                sub_summary = sub_summary.append(predictor_summary)
                

        sub_summary.to_csv(f'{curr_dir}/results/mvpd/{exp}_{summary_type}_{analysis_type}.csv', index=False)