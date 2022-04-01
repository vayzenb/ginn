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
import itertools
import glob
import models


import deepdish as dd

scaler = T.Resize((224, 224))
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
to_tensor = T.ToTensor()

image_dir = "/user_data/vayzenbe/GitHub_Repos/ginn/Stim"

ModelType = ['Face','Object', 'Random']
model_arch
stim = ['vggface2_fbf', 'ImageNet_Objects']
cond =['Upright', 'Inverted']
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
        for ss in range(0, len(stim)):
            #loop through folders
            imFolders = next(os.walk(f"{image_dir}{stim[ss]}/val/"))[1]
            #Iterate through each image and extract activations
            
            for cc in cond:
                #Set up empty dictionary
                #This has more features than needed, but it will get pruned at the end
                #Maybe remove this for individual activation files
                Acts = {'Act' : np.zeros((30000, 1024)),'Label' : np.zeros((30000, 1))}
                imNum = 0
                n=0
                for ii in range(0, len(imFolders)):
                    #load images in that folder
                    #THey are annoying in two different formats between VGG (.jpg) and ImageNet (.JPEG)
                    imFiles = [os.path.basename(x) for x in glob.glob(f"{image_dir}{stim[ss]}/val/{imFolders[ii]}/*.jpg")]
                    imFiles.extend([os.path.basename(x) for x in glob.glob(f"{image_dir}{stim[ss]}/val/{imFolders[ii]}/*.JPEG")])

                    for jj in range(0, len(imFiles)):
                        IM = Image.open(f"{image_dir}{stim[ss]}/val/{imFolders[ii]}/{imFiles[jj]}").convert("RGB")
                        
                        if cc == 'Inverted':
                            IM = IM.rotate(180) # rotate to invert it
                        
                        IM = image_loader(IM) #Convert to tensor
                        _model_feats = []
                        model(IM)
                        Acts['Act'][n,:] = _model_feats[0][0]
                        Acts['Label'][n] = imNum
                        n = n + 1
                        
                    print(ModelType[mm], stim[ss],cc, imNum, n)    
                    imNum = imNum + 1

                    if imNum == 10:
                        break

            
                #Remove all unsused rows and save
                Acts['Act'] = Acts['Act'][0:n,:]
                Acts['Label'] = Acts['Label'][0:n]
                
                dd.io.save(f"Activations/{ModelType[mm]}_{stim[ss]}_{cc}.h5", Acts)
                print(ModelType[mm], stim[ss], cc, 'Saved')