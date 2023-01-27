curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}/fmri')
sys.path.insert(1, f'{curr_dir}')
import indiv_mvpd
import analysis_funcs
import pandas as pd
import numpy as np
import ginn_params as params
import pdb


human_predict = True
model_predict = True
suf = '_indiv_mvpd'
'''
model predictors
'''
model_arch = ['cornet_z_sl']
train_type = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer = ['aIT']

exp = params.exp
vid = params.vid
fix_tr = params.fix_tr


'''
neural predictors
'''
ages = ['adult']
rois = ['LO','FFA','A1']


n_feats = [25,50,100,200]
n_feats = [25]

if human_predict == True:

    predictor_dir = f'/lab_data/behrmannlab/vlad/ginn/fmri/{exp}/derivatives/group_func'
    summary_type = 'human'

    for n_feat in n_feats:
        sub_summary = pd.DataFrame(columns=['sub','age', 'roi','corr','seed_age', 'seed_roi'])
        for age in ages:
            for rr in rois:
                for lr in ['l','r']:
                    roi = f'{lr}{rr}'
                    print(f'predicting using {age} and {roi}...', n_feat)
                    predictor_ts = np.load(f'{predictor_dir}/srm_{roi}_{age}_{n_feat}.npy')
                    predictor_ts = predictor_ts[:,fix_tr:]
                    predictor_ts = np.transpose(predictor_ts)
                    predictor_summary = indiv_mvpd.predict_indvidual(predictor_ts)
                    
                    predictor_summary['seed_age'] = age
                    predictor_summary['seed_roi'] = roi

                    sub_summary = sub_summary.append(predictor_summary)
                    
                    
        
        sub_summary.to_csv(f'{curr_dir}/results/mvpd/human_{exp}_{summary_type}_summary_{n_feat}_{suf}.csv', index=False)

if model_predict == True:
    predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
    summary_type = 'model'
    sub_summary = pd.DataFrame(columns=['sub','age', 'roi','r2','architecture', 'train_type', 'layer'])
    for mt in model_arch:
        for tt in train_type:
            for ll in layer:
                print(f'predicting using {mt} {tt} {ll}...')
                predictor_ts = np.load(f'{predictor_dir}/{mt}_{tt}_{ll}_{vid}_ts.npy')
                #predictor_ts = predictor_ts[fix_tr:,:]
                #pdb.set_trace()

                n_comps = analysis_funcs.calc_pc_n(analysis_funcs.extract_pc(predictor_ts), 0.9)
                pca = analysis_funcs.extract_pc(predictor_ts, n_comps)
                predictor_ts = pca.transform(predictor_ts)
                
                #predictor_ts = np.transpose(predictor_ts)
                predictor_summary = indiv_mvpd.predict_indvidual(predictor_ts)
                
                
                predictor_summary['architecture'] = mt
                predictor_summary['train_type'] = tt
                predictor_summary['layer'] = ll

                sub_summary = sub_summary.append(predictor_summary)
                

        sub_summary.to_csv(f'{curr_dir}/results/mvpd/{exp}_{summary_type}_{mt}_summary{suf}.csv', index=False)
