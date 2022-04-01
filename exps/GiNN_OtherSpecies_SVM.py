# -*- coding: utf-8 -*-
"""
Conducts SVM analysis for the perceptual narrowing experiment on object, face, and random models

Tests how well each model can categorize, objects, human faces, and monkey faces

Created on Sun Dec 29 14:43:29 2019

@author: vayze
"""


from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
import random

from skimage import io
import deepdish as dd

from itertools import chain

#Number of data splits and folds
ns = 20


ModelType = ['Face','Object', 'Random']
stim = ['FaceGen', 'Monkey', 'Human']
stimCat = ['FaceGen', 'Monkey', 'Human']

for mm in range(0, len(ModelType)):
    for ss in range(0, len(stim)):
        allActs = dd.io.load(f"Activations/{ModelType[mm]}_{stim[ss]}_OtherSpecies.h5")
        allActs['Label'] = allActs['Label'].astype(int) #convert labels to int
        sss = StratifiedShuffleSplit(n_splits=ns,test_size=0.4)

        CNN_Acc = np.empty([ns,3], dtype=object)
            
        tempScore = []
        X = allActs['Act']
        y = allActs['Label'].flatten()        
        print("starting train loop")
        n=0
        n= 0
        
        for train_index, test_index in sss.split(X, y):
        
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
            currScore = clf.score(X_test, y_test)

            print(n, ModelType[mm], stim[ss], currScore)
            CNN_Acc[n,0] = ModelType[mm]
            CNN_Acc[n,1] = stimCat[ss]
            CNN_Acc[n,2] = currScore

            tempScore.append(currScore) 
            n = n + 1

        score = np.mean(tempScore)
            
        np.savetxt(f"Results/{ModelType[mm]}_{stimCat[ss]}_OtherSpecies.csv", CNN_Acc, delimiter=',', fmt= '%s')            
        print(ModelType[mm], stimCat[ss], 'Avg:', score)
