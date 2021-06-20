# -*- coding: utf-8 -*-
"""
Extracts activations from penultimate layer from object, face, and random models
to object, human, and monkey stimuli

Created on Thu Dec 26 18:25:51 2019

@author: VAYZENB
"""


import os
#os.chdir('C:/Users/vayze/Desktop/GitHub Repos/GiNN/')

import sys
sys.path.insert(1, '/home/vayzenbe/GitHub_Repos/GiNN/Models/')

import torch as nn
import torch
import torchvision
from torch.autograd import Variable
from PIL import Image
from skimage import io, transform
import numpy as np
import torchvision.transforms as T
import itertools
import glob
import cornet
import deepdish as dd

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()


ModelType = ['Face','Object', 'Random']
stim = ['FaceGen', 'Monkey', 'Human']
epoch = 30
layer = 'decoder'
sublayer = 'avgpool'

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


for mm in range(0, len(ModelType)):

    #Load model
    model = getattr(cornet, 'cornet_z')
    model = model(pretrained=False, map_location='gpu')

    #If face or object, load face or object weights. else leave are random
    if ModelType[mm] == 'Face' or ModelType[mm] == 'Object':
        checkpoint = torch.load(f"Models/data/CorNet_{ModelType[mm]}_{epoch}.pt")
        
        model.load_state_dict(checkpoint)    

    #set up hook to specified layer
    try:
        m = model.module
    except:
        m = model
    model_layer = getattr(getattr(m, layer), sublayer)
    model_layer.register_forward_hook(_store_feats)

    model.to(device)
    #set to eval mode
    model.eval()
    with torch.no_grad():
        ##run all images through the model
        for ss in range(0, len(stim)):
            #Set up empty dictionary
            #This has more features than needed, but it will get pruned at the end
            #Maybe remove this for individual activation files
            Acts = {'Act' : np.zeros((30000, 1024)),'Label' : np.zeros((30000, 1))}
            imFolders = next(os.walk('Stim/' + stim[ss] + '/'))[1]

            #Iterate through each image and extract activations
            n=0
            imNum = 0
            for ii in range(0, len(imFolders)):
            #for ii in range(0,3):
                imFiles = [os.path.basename(x) for x in glob.glob(f"Stim/{stim[ss]}/{imFolders[ii]}/*.jpg")]
                
                for jj in range(0, len(imFiles)):
                #for jj in range(0, 3):
                    IM = Image.open(f"Stim/{stim[ss]}/{imFolders[ii]}/{imFiles[jj]}").convert("RGB")

                    IM = image_loader(IM)
                    _model_feats = []
                    model(IM)
                    Acts['Act'][n,:] = _model_feats[0][0]
                    Acts['Label'][n] = imNum
                    n = n + 1

                   # print(ModelType[mm], stim[ss], imNum, imFiles[jj], n)
                    
                print(ModelType[mm], stim[ss], imNum, n) 
                imNum =imNum + 1

                    #Remove all unsused rows and save
            Acts['Act'] = Acts['Act'][0:n,:]
            Acts['Label'] = Acts['Label'][0:n]

            dd.io.save(f"Activations/{ModelType[mm]}_{stim[ss]}_OtherSpecies.h5", Acts)
            print(ModelType[mm], stim[ss], 'saved')

        






















