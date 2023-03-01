"""
Find optimal image for an encoding model
"""
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')
sys.path.insert(1, f'{curr_dir}/fmri')

import pandas as pd
import numpy as np
import pdb
import ginn_params as params
import mean_ts_movie_crossval as predict_script
import analysis_funcs

from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
import pickle

predict_ts = predict_script.predict_ts

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, fmri_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)
#find the optimal layer/PCs for region using full dataset
# 
#loop through layers
# 
model_arch = 'cornet_z_sl'
model_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
layers = ['V1','V2','V4','pIT','aIT', 'decoder']

model_dir = '/lab_data/behrmannlab/vlad/ginn/modelling'
summary_type = 'model'
rois = ['FFA','lFFA','rFFA']

#loop through model and layers
for model in model_types:
    for layer in layers:

        print(f'predicting using {model_arch} {model} {layer}...')
        predictor_ts = np.load(f'{model_dir}/model_ts/{model_arch}_{model}_{layer}_{vid}_ts.npy')

        #standardize predictor_ts
        predictor_ts = stats.zscore(predictor_ts, axis=0)

        #convert nans to 0
        predictor_ts[np.isnan(predictor_ts)] = 0

        pca = analysis_funcs.extract_pc(predictor_ts)
        predictor_comps = pca.transform(predictor_ts)

        #standardize predictor_ts
        predictor_comps = stats.zscore(predictor_comps, axis=0)

        for age in ages:
            curr_subs = sub_list[sub_list['AgeGroup'] == age]

            for roi in rois:
                
                #extract roi data
                roi_data = predict_script.extract_roi_data(curr_subs, f'{roi}')
                roi_data = np.asarray(roi_data)
                

                #standardize roi data
                roi_data = stats.zscore(roi_data, axis=0)
                #average roi data
                roi_mean = np.mean(roi_data, axis=0)

                #find optimal pc
                max_pc, score = predict_script.find_optimal_pc(roi_data, predictor_comps)
                

                #create encoding model using optimal pc
                clf =Ridge()
                clf.fit(predictor_comps[:,0:max_pc], roi_mean)

                #save encoding model
                pickle.dump(clf, open(f'{model_dir}/encoding_models/{model_arch}_{model}_{layer}_{roi}_{age}_ridge.pkl', 'wb'))

                #save optimal pc
                np.save(f'{model_dir}/encoding_models/{model_arch}_{model}_{layer}_{roi}_{age}_optimal_pc.npy', max_pc)

                print(f'{model_arch} {model} {layer}...', score)