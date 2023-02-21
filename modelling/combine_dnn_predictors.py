'''
Create combined DNN predictor

Takes most predictive layers from each DNN and combines them into a single
'''
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys

sys.path.insert(1, f'{curr_dir}')

import warnings
import os, argparse
from matplotlib.pyplot import subplot

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb

from sklearn.decomposition import PCA


from scipy import stats
import ginn_params as params
import random
import analysis_funcs
print('libraries loaded')

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, file_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)
predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'

model_arch = 'cornet_z_sl'
model_types = [['imagenet_noface',  'vggface'],['imagenet_noface', 'imagenet_vggface',  'vggface'],
               ['imagenet_noface',  'vggface'],['imagenet_noface', 'imagenet_vggface',  'vggface']]
layers = [['V2','pIT'],['V2','V1','pIT'],
          ['V2','aIT'],['V2','pIT','aIT']]


for mt, models in enumerate(model_types):
    model_data = []
    model_len = []
    #load data from all models
    #run PCA on each model
    #add to model_data
    for m in models:
        layer = layers[mt][models.index(m)]
        
        #load model data 
        predictor_ts = np.load(f'{predictor_dir}/{model_arch}_{m}_{layer}_{vid}_ts.npy')
        print(f'{predictor_dir}/{model_arch}_{m}_{layer}_{vid}_ts.npy')
        #standardize predictor_ts
        predictor_ts = stats.zscore(predictor_ts, axis=0)

        #convert nans to 0
        predictor_ts[np.isnan(predictor_ts)] = 0


        pca = analysis_funcs.extract_pc(predictor_ts)
        predictor_comps = pca.transform(predictor_ts)
        #standardize predictor_ts
        predictor_comps = stats.zscore(predictor_comps, axis=0)

        model_data.append(predictor_comps)
        model_len.append(predictor_comps.shape[1])


    #combine model data in alternating order
    min_pcs = min(model_len)
    max_pcs  = max(model_len)

    
    #create alternating order
    combined_data = np.zeros((predictor_ts.shape[0], min_pcs*len(models)))
    for i in range(0, min_pcs):
        combined_data[:,i] = model_data[0][:,i]
        combined_data[:,i+1] = model_data[1][:,i]


    #save combined predictor
    np.save(f'{predictor_dir}/{model_arch}_{"_".join(models)}_{"_".join(layers[mt])}_{vid}_ts.npy', combined_data)


