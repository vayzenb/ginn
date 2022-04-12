# -*- coding: utf-8 -*-
"""
Tests whether face, object, random models are better at identifying faces when they are upright vs. inverted

Created on Thu Feb  6 18:32:30 2020

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
ns = 10
nr=5

#number of classes to test
nCat = 30


ModelType = ['Face','Object', 'Random']
stim = ['vggface2_fbf', 'ImageNet_Objects']
stimCat = ['Face', 'Object']
cond =['Upright', 'Inverted']

for mm in range(0, len(ModelType)):
    for ss in range(0, len(stim)):
        for cc in cond:
            allActs = dd.io.load(f"Activations/{ModelType[mm]}_{stim[ss]}_{cc}.h5")
            allActs['Label'] = allActs['Label'].astype(int) #convert labels to int

            CNN_Acc = np.empty([ns*nr,4], dtype=object)
            
            tempScore = []
                    
            sss = StratifiedShuffleSplit(n_splits=ns,test_size=0.2)
            print('about to start train loop')
            n= 0
            for nRep in range(0,nr):
                rInd = random.sample(range(0, int(np.max(allActs['Label']))), nCat) #randomly select some classes
            
                #iterate through random classes and pull out relevant values
                X = np.empty([0, allActs['Act'].shape[1]])
                y = np.array([])
                for ind in rInd: 
                    idx = (allActs['Label'] == ind)
                    X = np.append(X, allActs['Act'][idx.flatten(),:],axis = 0)
                    y = np.append(y, allActs['Label'][allActs['Label'] == ind])
                for train_index, test_index in sss.split(X, y):
                
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
                    currScore = clf.score(X_test, y_test)
                    print(n, ModelType[mm], stim[ss], cc, currScore)
                    CNN_Acc[n,0] = ModelType[mm]
                    CNN_Acc[n,1] = stimCat[ss]
                    CNN_Acc[n,2] = cc
                    CNN_Acc[n,3] = currScore

                    tempScore.append(currScore) 
                    n = n + 1

            score = np.mean(tempScore)
                
            np.savetxt(f"Results/{ModelType[mm]}_{stimCat[ss]}_{cc}.csv", CNN_Acc, delimiter=',', fmt= '%s')            
            print(ModelType[mm], stim[ss], cc, 'Avg:', score)
            