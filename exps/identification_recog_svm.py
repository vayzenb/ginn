# -*- coding: utf-8 -*-
"""
Extracts activations from penultimate layer from object, face, and random models
to human and inverted human faces

Created on Thu Feb  6 14:02:03 2020

@author: vayze
"""


import os
#os.chdir('C:/Users/vayze/Desktop/GitHub Repos/GiNN/')

import sys
sys.path.insert(1, '/user_data/vayzenbe/GitHub_Repos/lemniscate.pytorch/')

import torch as nn
import torch
import torchvision
from torch.autograd import Variable
from PIL import Image
from skimage import io, transform
import numpy as np
import torchvision.transforms as T
from torchvision import transforms
import itertools
import glob
import models
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd


import deepdish as dd
import pdb

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
test_dir = "/user_data/vayzenbe/image_sets/"
#test_dir = "/lab_data/plautlab/imagesets/"
weights_dir = "/user_data/vayzenbe/GitHub_Repos/lemniscate.pytorch/results"

train_type = ['random', 'imagenet_objects']
model_arch = ['cornet_z']
test_stim = ['objects']

test_cond =['upright', 'inverted']
epoch = 30
layer = 'decoder'
sublayer = 'avgpool'
suf = '_unsupervised'
splits = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")







#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image_name = Variable(normalize(to_tensor(scaler(image_name))).unsqueeze(0))
    return image_name

def _store_feats(layer, inp, output):
    """An ugly but effective way of accessing intermediate model features
    """   
    output = output.cpu().detach().numpy()
    _model_feats.append(np.reshape(output, (len(output), -1)))

def load_model(model_arch, train_type):
    print('loading model...')
    #Load model
    model = models.__dict__[model_arch](low_dim=128)
    model = torch.nn.DataParallel(model).cuda()
    
    #If face or object, load face or object weights. else leave are random
    if train_type != 'random':
        checkpoint = torch.load(f"{weights_dir}/{model_arch}_{train_type}_checkpoint.pth.tar")
        
        model.load_state_dict(checkpoint['state_dict']) 
    

    model.eval()

    return model

def extract_acts(model, image_dir, cond):
    print('extracting features...')
    cond = ['upright','inverted']

    #set up hook to specified layer
    def _store_feats(layer, inp, output):
        """An ugly but effective way of accessing intermediate model features
        """
        output = output.cpu().numpy()
        _model_feats.append(np.reshape(output, (len(output), -1)))

    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)
    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)
    #loop through folders
    imFolders = next(os.walk(image_dir))[1]

    #Iterate through each image and extract activations

    #Set up empty dictionary
    #This has more features than needed, but it will get pruned at the end
    #Maybe remove this for individual activation files
    acts = {'feats' : np.zeros((30000, 1024)),'label' : np.zeros((30000, 1))}
    imNum = 0
    n=0

    if cond == 'inverted':
        transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])
    else:
        #Transformations for ImageNet
        transform = transforms.Compose([
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])])


    
    test_dataset = torchvision.datasets.ImageFolder(image_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers = 4, pin_memory=True)
    


    with torch.no_grad():
        for data, label in testloader:
            # move tensors to GPU if CUDA is available
            data, target = data.cuda(), label.cuda()
            
            _model_feats = []
            model(data)
            #output = model(data)

            labels = label.cpu().numpy()

            if n == 0:
                acts = {'feats' : _model_feats[0],'label' : labels}
            else:
                acts['feats'] = np.append(acts['feats'], _model_feats[0],axis = 0)
                acts['label'] = np.append(acts['label'], labels,axis = 0)
            
            n = n + 1

                            
        
    

    return acts

def start_classification(acts, model_info):
    print('Running classification...')
    clf = svm.SVC(kernel='linear', C=1)
    sss = StratifiedShuffleSplit(n_splits=splits,test_size=0.2)
    X = acts['feats']
    y = np.ravel(acts['label']).astype(int)

    cat_summary = pd.DataFrame(columns = ['model_arch', 'train_type', 'test_stim','test_cond', 'fold_n','acc'])
    fold_n = 1
    for train_index, test_index in sss.split(X, y):
        
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        clf.fit(X_train, y_train)
        curr_acc = clf.score(X_test, y_test)
        #pdb.set_trace()
        curr_data = pd.Series(model_info + [fold_n, curr_acc], index = cat_summary.columns)
        cat_summary = cat_summary.append(curr_data, ignore_index=True)
        fold_n +=1

    print(model_info,cat_summary['acc'].mean())
    return cat_summary

    



for mm in enumerate(model_arch):   
    for trt in enumerate(train_type):
        model = load_model(mm[1], trt[1])
        


        for ts in test_stim:
            test_ims = f'{test_dir}/{ts}'

            for cc in test_cond:
                file_name = f'{mm[1]}_{trt[1]}_{ts}_{cc}{suf}'
                acts = extract_acts(model, test_ims, cc)

                dd.io.save(f"{curr_dir}/activations/{file_name}.h5", acts)
                model_info = [mm[1], trt[1], ts, cc]
                cat_summary = start_classification(acts, model_info)
                
                cat_summary.to_csv(f"{curr_dir}/results/{file_name}.csv", index=False)
                #print(mm[1], ts,cc, 'Saved')