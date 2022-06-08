curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}/fmri')
import child_mvpd
import pandas as pd
import numpy as np
import pdb



#

srm_predict = False
model_predict = True

'''
model predictors
'''
model_arch = ['cornet_z_cl','cornet_z_sl']
train_type = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layer = ['aIT','pIT']
vid = 'DM-clip'
'''
neural predictors
'''
ages = [18]
rois = ['LO','FFA','OFA']


n_feats = [25,50,100,200]

if srm_predict == True:

    predictor_dir = '/lab_data/behrmannlab/scratch/vlad/ginn/fmri/hbn/derivatives/group_func'
    summary_type = 'srm'

    for n_feat in n_feats:
        sub_summary = pd.DataFrame(columns=['sub','age', 'roi','r2','seed_age', 'seed_roi'])
        for age in ages:
            for rr in rois:
                for lr in ['l','r']:
                    roi = f'{lr}{rr}'
                    print(f'predicting using {age} and {roi}...', n_feat)
                    predictor_ts = np.load(f'{predictor_dir}/srm_{roi}_{age}_{n_feat}.npy')
                    
                    predictor_ts = np.transpose(predictor_ts)
                    predictor_summary = child_mvpd.calc_mvpd_r2(predictor_ts)
                    
                    predictor_summary['seed_age'] = age
                    predictor_summary['seed_roi'] = roi

                    sub_summary = sub_summary.append(predictor_summary)
                    
                    
        
        sub_summary.to_csv(f'{curr_dir}/results/mvpd/{summary_type}_summary_{n_feat}.csv')

if model_predict == True:
    predictor_dir = '/lab_data/behrmannlab/vlad/ginn/model_ts'
    summary_type = 'model'
    sub_summary = pd.DataFrame(columns=['sub','age', 'roi','r2','architecture', 'train_type', 'layer'])
    for mt in model_arch:
        for tt in train_type:
            for ll in layer:
                print(f'predicting using {mt} {tt} {ll}...')
                predictor_ts = np.load(f'{predictor_dir}/{mt}_{tt}_{ll}_{vid}_ts.npy')
                
                #predictor_ts = np.transpose(predictor_ts)
                predictor_summary = child_mvpd.calc_mvpd_r2(predictor_ts)
                
                
                predictor_summary['architecture'] = mt
                predictor_summary['train_type'] = tt
                predictor_summary['layer'] = ll

                sub_summary = sub_summary.append(predictor_summary)
                

        sub_summary.to_csv(f'{curr_dir}/results/mvpd/{summary_type}_{mt}_summary.csv')
