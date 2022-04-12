"""
Extracts activations from penultimate layer from object, face, and random models
to human and inverted human faces

Created on Thu Feb  6 14:02:03 2020

@author: vayze
"""


import os
#os.chdir('C:/Users/vayze/Desktop/GitHub Repos/GiNN/')

import sys


import torch as nn
import torch
import torchvision
from torch.autograd import Variable
from PIL import Image
from skimage import io, transform
import numpy as np
from torchvision import transforms
import itertools
import glob
import models
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import load_stim


import deepdish as dd
import pdb
import warnings
warnings.filterwarnings("ignore")

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()

curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
test_dir = "/user_data/vayzenbe/image_sets/"
act_dir = '/lab_data/behrmannlab/vlad/ginn/activations'
#test_dir = "/lab_data/plautlab/imagesets/"
weights_dir = '/lab_data/behrmannlab/vlad/ginn/model_weights'

train_type = ['random','imagenet_noface', 'imagenet_oneface',
'imagenet_vggface', 'vggface_oneobject', 'vggface']
model_arch = ['cornet_z']
test_stim = ['objects', 'faces']

exp = 'classify'

os.makedirs(f'{act_dir}/{exp}', exist_ok = True)
os.makedirs(f"{curr_dir}/results/{exp}/", exist_ok= True)

seed = 1

test_only = False

test_cond =['upright','inverted']
epoch = 30
layer = 'decoder'
sublayer = 'linear'
suf = '_unsupervised'
splits = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image_name = Variable(normalize(to_tensor(scaler(image_name))).unsqueeze(0))
    return image_name


def load_model(model_arch, train_type):
    print('loading model...')
    #Load model
    model = models.__dict__[model_arch](low_dim=128)
    model = torch.nn.DataParallel(model).cuda()
    
    #If face or object, load face or object weights. else leave are random
    if train_type != 'random':
        checkpoint = torch.load(f"{weights_dir}/{model_arch}_{train_type}_best_{seed}.pth.tar")
        
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



    #Iterate through each image and extract activations

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
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 4, pin_memory=True)
    


    with torch.no_grad():
        for data, label in testloader:
            # move tensors to GPU if CUDA is available
            data, target = data.cuda(), label.cuda()
            
            _model_feats = []
            model(data)
            #output = model(data)
            
            out = np.vstack(_model_feats)

            labels = label.cpu().numpy()

            if n == 0:
                acts = {'feats' : out,'label' : labels}
            else:
                acts['feats'] = np.append(acts['feats'], out,axis = 0)
                acts['label'] = np.append(acts['label'], labels,axis = 0)
            
            n = n + 1

    return acts