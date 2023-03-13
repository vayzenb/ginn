"""
Calculate correlation between models
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
from joblib import dump, load
import random

#predict_ts = predict_script.predict_ts

exp = 'aeronaut'

model_arch = 'cornet_z_sl'
train_type1 = 'imagenet_noface'
train_type2 = 'vggface'
layer1 = 'decoder'
layer2 = 'aIT'
rois = ['FFA']

predictor_dir = '/lab_data/behrmannlab/vlad/ginn/modelling/model_ts'
results_dir = f'{curr_dir}/results/mean_ts'

alpha = .05

folds = 50 #CHANGE

summary_cols = ['model_arch','train_type1','layer1','train_type2','layer2','age','roi', 'corr','se', 'ci_low','ci_high']

def setup_predictors(vid, model_arch, train_type, layer):
    predictor_ts = np.load(f'{predictor_dir}/{model_arch}_{train_type}_{layer}_{vid}_ts.npy')

    #standardize predictor_ts
    predictor_ts = stats.zscore(predictor_ts, axis=0)

    #convert nans to 0
    predictor_ts[np.isnan(predictor_ts)] = 0

    pca = analysis_funcs.extract_pc(predictor_ts)
    predictor_comps = pca.transform(predictor_ts)

    #standardize predictor_ts
    predictor_comps = stats.zscore(predictor_comps, axis=0)

    return predictor_comps

def fit_ts(seed_train1,seed_test1,seed_train2,seed_test2, train_data, test_data):
    #standardize all variables
    seed_train1 = stats.zscore(seed_train1, axis=0)
    seed_test1 = stats.zscore(seed_test1, axis=0)
    seed_train2 = stats.zscore(seed_train2, axis=0)
    seed_test2 = stats.zscore(seed_test2, axis=0)
    train_data = stats.zscore(train_data, axis=0)
    test_data = stats.zscore(test_data, axis=0)

    #fit model
    clf = Ridge()
    clf.fit(seed_train1, train_data)
    pred1 = clf.predict(seed_test1)

    #fit second model
    clf = Ridge()
    clf.fit(seed_train2, train_data)
    pred2 = clf.predict(seed_test2)

    score = np.corrcoef(pred1.flatten(),pred2.flatten())[0,1]

    return score

def cross_val(roi_data,predictor_ts1, predictor_ts2):
    
    print('running cross validation...')
    roi_data = np.asanyarray(roi_data)
    cv_ind = np.arange(0,len(roi_data)).tolist()
    
    #split into folds
    score = []
    for fold in range(0,folds):

        #shuffle cv_ind use fold as the same random seed
        random.Random(fold).shuffle(cv_ind)
        
        #split into train and test for hyperparameter tuning
        #use train data to find how many PCs to use
        hp_train = roi_data[cv_ind[0:int(len(cv_ind)/2)]]
        hp_test = roi_data[cv_ind[int(len(cv_ind)/2):]]

        #find optimal PCs
        optimal_pc1, _ = predict_script.find_optimal_pc(hp_train, predictor_ts1)
        optimal_pc2, _ = predict_script.find_optimal_pc(hp_train, predictor_ts2)

        seed_ts1 = predictor_ts1[:,0:optimal_pc1]
        seed_ts2 = predictor_ts2[:,0:optimal_pc2]


        roi_mean = np.mean(hp_test,axis=0)        

        split_score = []
        for movie_half in range(0,2):
            
            if movie_half == 0:
                seed_train1 = seed_ts1[0:int(len(seed_ts1)/2),:]
                seed_test1 = seed_ts1[int(len(seed_ts1)/2):,:]

                seed_train2 = seed_ts2[0:int(len(seed_ts2)/2),:]
                seed_test2 = seed_ts2[int(len(seed_ts2)/2):,:]

                
                target_train = roi_mean[0:int(len(roi_mean)/2)]
                target_test = roi_mean[int(len(roi_mean)/2):]


            elif movie_half == 1:
                seed_train1 = seed_ts1[int(len(seed_ts1)/2):,:]
                seed_test1 = seed_ts1[0:int(len(seed_ts1)/2),:]

                seed_train2 = seed_ts2[int(len(seed_ts2)/2):,:]
                seed_test2 = seed_ts2[0:int(len(seed_ts2)/2),:]

                target_train = roi_mean[int(len(roi_mean)/2):]
                target_test = roi_mean[0:int(len(roi_mean)/2)]

                hp_train = hp_train[int(len(hp_train)/2):,:]
                hp_test = hp_test[int(len(hp_test)/2):,:]

            
            curr_score = fit_ts(seed_train1,seed_test1,seed_train2, seed_test2, target_train, target_test)

            split_score.append(curr_score)

        score.append(np.mean(split_score))
        

    mean_score = np.mean(score)
    #caluclate 95% confidence interval
    ci_low = np.percentile(score, alpha*100)
    ci_high= np.percentile(score, 100-alpha*100)

    se = np.std(score)/np.sqrt(folds)


   


    return mean_score, se, ci_low, ci_high

def predict_loop(exp,  model_arch, train_type1,train_type2, layer1,layer2):
    #load exp params
    study_dir,subj_dir, sub_list, vid, fmri_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)

    #load first model and layer
    predictor_ts1 = setup_predictors(vid,  model_arch, train_type1, layer1)

    #load second model and layer
    predictor_ts2 = setup_predictors(vid,  model_arch, train_type2, layer2)
    
    sub_summary = pd.DataFrame(columns = summary_cols)
    for age in ages:
        curr_subs = sub_list[sub_list['AgeGroup'] == age]
        
        roi_summary = pd.DataFrame(columns = summary_cols)
        for roi in rois:
            print(f'predicting {roi} for {age} {model_arch} {train_type1} {layer1} {train_type2} {layer2}...')
            #extract roi data
            roi_data = predict_script.extract_roi_data(curr_subs, f'{roi}')
            roi_data = np.asarray(roi_data)
            
            #standardize roi data
            roi_data = stats.zscore(roi_data, axis=0)

            score, score_se,ci_low, ci_high = cross_val(roi_data,predictor_ts1, predictor_ts2)

            
            #concat to summary
            sub_summary = sub_summary.append(pd.DataFrame([[model_arch, train_type1, layer1, train_type2, layer2, roi, age, score, score_se, ci_low, ci_high]], columns = summary_cols))
            
    #save summary
    sub_summary.to_csv(f'{results_dir}/{exp}_{model_arch}_{train_type1}_{layer1}_{train_type2}_{layer2}_model_corr.csv', index=False)


 


predict_loop(exp,  model_arch, train_type1,train_type2, layer1,layer2)
