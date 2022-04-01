# -*- coding: utf-8 -*-
"""
Created on Sun May 10 2020

Tests whether schematic upright/inverted schematic face images can be categorized after a classifier is trained on upright object/faces


@author: VAYZENB
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

image_dir = "/Stim/"

ModelType = ['Face','Object', 'Random']
train_stim = 'schematic'
test_stim = 'front_face'
cond =['Upright', 'Inverted']

epoch = 30
layer = 'decoder'
sublayer = 'avgpool'


n = 0
for mm in range(0, len(ModelType)):

    #Load SVM training stim (schematic)
    allActs = dd.io.load(f"Activations/{ModelType[mm]}_{train_stim}.h5")
    allActs['Label'] = allActs['Label'].astype(int) #convert labels to int
    X_train = allActs['Act']
    y_train = allActs['Label'].flatten()  

    #load testing stim
    #Load SVM training stim (schematic)
    allActs = dd.io.load(f"Activations/{ModelType[mm]}_{test_stim}.h5")
    allActs['Label'] = allActs['Label'].astype(int) #convert labels to int
    X_test = allActs['Act']
    y_test = allActs['Label'].flatten()  

    CNN_Acc = np.empty([6,4], dtype=object)
    
    tempScore = []
    
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    
    currScore = clf.score(X_test, y_test)
    print(n, ModelType[mm], train_stim, test_stim, currScore)
    CNN_Acc[n,0] = ModelType[mm]
    CNN_Acc[n,1] = train_stim
    CNN_Acc[n,2] = test_stim
    CNN_Acc[n,3] = currScore

    tempScore.append(currScore) 
    n = n + 1

    np.savetxt(f"Results/{ModelType[mm]}_TopHeavy.csv", CNN_Acc, delimiter=',', fmt= '%s')            
