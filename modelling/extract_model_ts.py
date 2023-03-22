# %%
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
import os, argparse
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps,  ImageFilter
from itertools import chain
import pandas as pd
import numpy as np
import models
import model_funcs
import matplotlib.pyplot as plt
from statistics import mean

import load_stim
from scipy import stats, spatial
from sklearn.decomposition import PCA
from scipy.stats import gamma
from glob import glob as glob

import ginn_params as params

import pdb

print('libraries loaded')

#if you've already (correctly) extracted the activations, just load them; set to True 
acts_extracted = False
exp = 'aeronaut'

'''
folder params
'''
study_dir,subj_dir, sub_list, vid, fmri_suf, start_trs,end_trs, data_dir, vols, tr, fps, bin_size, ages= params.load_params(exp)

stim_dir = f"{curr_dir}/stim/fmri_videos/frames"
weights_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/model_weights"
out_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/model_ts"


#training info

model_arch = 'cornet_z_sl'
'''
set model params
'''

train_type = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']


train_dir = f'/lab_data/behrmannlab/image_sets/'


seed = 1
pca_perc = .90
epochs = [0, 1, 5, 10, 15, 20, 25, 30]

global layer
global sublayer

#load first arg as train_type
if len(sys.argv) > 1:
    train_type = sys.argv[1]
    layer = sys.argv[2]
    sublayer = sys.argv[3]
    print(f'extracting for {layer} and {sublayer}')


transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

def load_model(model_arch, train_type,n_feats=128):
    print('loading model...')
    #Load model
    model = models.__dict__[model_arch](n_feats)
    
    
    
    #If face or object, load face or object weights. else leave are random
    if train_type != 'random':
        checkpoint = torch.load(f"{weights_dir}/{model_arch}_{train_type}_best_{seed}.pth.tar")
        try:    
            model.load_state_dict(checkpoint['state_dict']) 
            model = torch.nn.DataParallel(model).cuda()
        except:
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(checkpoint['state_dict'])
    else:
        model = torch.nn.DataParallel(model).cuda()

        
    model.eval()

    return model


def extract_acts(model, image_dir):
    print('extracting features...')
    

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

    
    #Transformations for ImageNet
    transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    
    test_dataset = load_stim.load_stim(image_dir, transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers = 1, pin_memory=False)
    


    with torch.no_grad():
        for data, _ in testloader:
            # move tensors to GPU if CUDA is available
            
            data= data.cuda()
            
            _model_feats = []
            model(data)
            #output = model(data)
            
            out = np.vstack(_model_feats)
            

            if n == 0:
                acts = out
            else:
                acts= np.append(acts, out,axis = 0)
                
            
            n = n + 1

    return acts




# %%
def down_sample(data):
    """Downsample data"""
    print('downsampling...')
    downsample_ts = np.empty((0, data.shape[1])) 

    
    #Bin frame data for TS
    for nn in range(0,len(data),bin_size):
        temp = data[nn:(nn+bin_size),:]

        downsample_ts = np.vstack((downsample_ts,np.mean(temp, axis=0)))

    #downsample_ts = downsample_ts[0:(vols-fix_tr),:] #extract only 168 volumes to match fmri data (credits of movie were cut)
    
    return downsample_ts

# %%

# %%
'''
Convolve with HRF using
Using double-gamma with 4-sec peak
'''

def hrf(data, tr):
    """ Return values for HRF at given times """
    times = np.arange(0, 30, tr)
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    hrf_at_trs = values / np.max(values) * 0.6 
    
    
    return np.convolve(data, hrf_at_trs)

# %%
def convolve_hrf(data):
    print('Convolving with HRF...')

    conv_ts =np.zeros((data.shape))
    for ii in range(0,data.shape[1]):
        temp = hrf(data[:,ii],tr)
        temp = temp[0:data.shape[0]]
        
        
        #temp = (temp - np.mean(temp))/np.std(temp)
        conv_ts[:,ii] = temp
        
    return conv_ts


# %%
'''
Model loop ver. 1
Conducts PCA on down-sample data
USE THIS ONE
'''

#set number of features in model
if model_arch == 'cornet_z_sl':

    if train_type == 'random':
        feats = 600
    else:
        feats = len(glob(f'{train_dir}/{train_type}/train/*'))
else:
    feats = 128

if acts_extracted == False:
    model = load_model(model_arch, train_type,feats)


print(f"Extracting timeseries for {model_arch}_{train_type}_{layer}...")


if acts_extracted == True:
    print('Acts loaded...')
    frame_acts = np.load(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_allframes.npy')
else:
    frame_acts = extract_acts(model, f'{stim_dir}/{vid}')

    #save full model timeseries (all frames)
    #np.save(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_allframes', frame_acts)





#downsample to scale of fmri
downsample_ts = down_sample(frame_acts)


#add burn volumes to beginning and end of timeseries as needed
if start_trs > 0:
    downsample_ts = np.vstack((np.zeros((start_trs,downsample_ts.shape[1])), downsample_ts))    

if end_trs > 0:
    downsample_ts = np.vstack((downsample_ts, np.zeros((end_trs,downsample_ts.shape[1]))))
    

#convolve to hrf
hrf_ts = convolve_hrf(downsample_ts)

final_ts = hrf_ts


np.save(f'{out_dir}/{model_arch}_{train_type}_{layer}_{vid}_ts', final_ts)

