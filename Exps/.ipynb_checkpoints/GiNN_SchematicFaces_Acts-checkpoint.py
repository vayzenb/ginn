# -*- coding: utf-8 -*-
"""
Extract acts for schematic and front view faces

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
gray_scale = T.Grayscale(num_output_channels=3)

image_dir = "Stim"

ModelType = ['Face','Object', 'Random']
stim = ['schematic', 'front_face']
cond =['Upright', 'Inverted']
epoch = 30
layer = 'decoder'
sublayer = 'avgpool'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Set image loader for model
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image_name = Variable(normalize(to_tensor(gray_scale(scaler(image_name)))).unsqueeze(0))
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
            imNum = 0 #Change imNum by conditions

            Acts = {'Act' : np.zeros((30000, 1024)),'Label' : np.zeros((30000, 1))}
            n=0            
            for cc in cond:
                #Set up empty dictionary
                #This has more features than needed, but it will get pruned at the end
                #Maybe remove this for individual activation files
                #load images in that folder
                #THey are annoying in two different formats between schamatic (.png) and face (.jpg)
                imFiles = [os.path.basename(x) for x in glob.glob(f"{image_dir}/{stim[ss]}/*.jpg")]
                imFiles.extend([os.path.basename(x) for x in glob.glob(f"{image_dir}/{stim[ss]}/*.png")])

                for jj in range(0, len(imFiles)):
                    IM = Image.open(f"{image_dir}/{stim[ss]}/{imFiles[jj]}").convert("RGB")
                    
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
        
            #Remove all unsused rows and save
            Acts['Act'] = Acts['Act'][0:n,:]
            Acts['Label'] = Acts['Label'][0:n]
            
            dd.io.save(f"Activations/{ModelType[mm]}_{stim[ss]}.h5", Acts)
            print(ModelType[mm], stim[ss], 'Saved')