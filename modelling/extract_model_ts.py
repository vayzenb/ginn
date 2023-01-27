# %%
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}/modelling')
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

import pdb

#if you've already (correctly) extracted the activations, just load them; set to True 
acts_extracted = False

'''
folder params
'''
vid = 'Aeronaut'
stim_dir = f"{curr_dir}/stim/fmri_videos/frames"
weights_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/model_weights"
out_dir = f"/lab_data/behrmannlab/vlad/ginn/modelling/model_ts"


#training info
model_arch = ['cornet_z_cl','cornet_z_sl']
model_arch = ['cornet_z_sl']
'''
set model params
'''

train_type = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
#train_type = ['vggface_oneobject', 'vggface', 'random']
#train_type = ['vggface_oneobject']

train_dir = f'/lab_data/behrmannlab/image_sets/'
#n_classes = len(glob(f'{args.data}/train/*'))
#layer =['aIT','pIT'] #set in descending order
layer_type = ['aIT']
#layer_type = ['decoder']
sublayer_type = 'output'
seed = 1
pca_perc = .90
epochs = [0, 1, 5, 10, 15, 20, 25, 30]



if vid == 'DM-clip':
    '''
    Despicable me (DM) video params
    '''
    vols = 750 #volumes in the scan
    tr = .8 #TR of scan
    fix_tr =0 #number of throwaway volumes at beginning
    fps = 30 # frame per second of video (how many rows go into 1 sec)
    bin_size = int(fps * tr) # get the bin size to average by multiplying the FPS by tr

elif vid == 'partly_cloudy':

    '''
    Pixar video params
    '''
    vols = 168 #volumes in the scan
    tr = 2 #TR of scan
    fix_tr =0 #number of throwaway volumes
    fps = 24 # frame per second of video (how many rows go into 1 sec)
    bin_size = fps * tr # get the bin size to average by multiplying the FPS by tr

elif vid == 'Aeronaut':

    '''
    Pixar video params
    '''
    vols = 90 #volumes in the scan
    tr = 2 #TR of scan
    fix_tr =0 #number of throwaway volumes
    fps = 24 # frame per second of video (how many rows go into 1 sec)
    bin_size = fps * tr # get the bin size to average by multiplying the FPS by tr


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
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers = 4, pin_memory=True)
    


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



# # %%
# def model_loop(model, loader):

#     im, label = next(iter(loader))

#     first_batch = True
#     for im, label in loader:
#         out = model_funcs.extract_acts(model, im)

#         if first_batch == True:
#             all_out = out
#             first_batch = False
#         else:
#             all_out = torch.cat((all_out, out), dim=0)

#     frame_acts = all_out.cpu().detach().numpy()
    
#     return frame_acts

# #idx = np.argwhere(np.all(all_out[..., :] == 0, axis=0))# find columns with only zeros
# #all_out = np.delete(all_out, idx, axis=1)
# #frame_ts = (all_out - np.mean(all_out,axis = 0))/np.std(all_out, axis = 0) # standardize the unit activations; this produces NaNs right now; figure out why
# #frame_ts = frame_ts[:, ~np.isnan(frame_ts).any(axis=0)]


# %%
def down_sample(data):
    """Downsample data"""
    downsample_ts = np.empty((0, data.shape[1])) 
    
    #Bin frame data for TS
    for nn in range(0,len(data),bin_size):
        temp = data[nn:(nn+bin_size),:]

        downsample_ts = np.vstack((downsample_ts,np.mean(temp, axis=0)))

    downsample_ts = downsample_ts[0:(vols-fix_tr),:] #extract only 168 volumes to match fmri data (credits of movie were cut)
    
    return downsample_ts

# %%
def extract_pc(data, n_components=None):

    """
    Extract principal components
    if n_components isn't set, it will extract all it can
    
    """
    #pdb.set_trace()
    pca = PCA(n_components = n_components)
    pca.fit(data)
    
    return pca

# %%
def calc_pc_n(pca, thresh):
    '''
    Calculate how many PCs are needed to explain X% of data
    
    pca - result of pca analysis
    thresh- threshold for how many components to keep
    '''

    explained_variance = pca.explained_variance_ratio_
    
    var = 0
    for n_comp, ev in enumerate(explained_variance):
        var += ev #add each PC's variance to current variance
        #print(evn, ev, var)

        if var >=thresh: #once variance > than thresh, stop
            break
    
    #plt.bar(range(len(explained_variance[0:n_comp])), explained_variance[0:n_comp], alpha=0.5, align='center')
    #plt.ylabel('Variance ratio')
    #plt.xlabel('Principal components')
    #plt.show()
    
    return n_comp

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

    conv_ts =np.zeros((data.shape))
    for ii in range(0,data.shape[1]):
        temp = hrf(data[:,ii],tr)
        temp = temp[0:(vols-fix_tr)] # only grab the first 168 volumes
        #temp = (temp - np.mean(temp))/np.std(temp)
        conv_ts[:,ii] = temp
        
    return conv_ts


# %%
'''
Model loop ver. 1
Conducts PCA on down-sample data
USE THIS ONE
'''
for mm in enumerate(model_arch):   
    for trt in enumerate(train_type):
        
        #set number of features in model
        if mm[1] == 'cornet_z_sl':

            if trt[1] == 'random':
                feats = 600
            else:
                feats = len(glob(f'{train_dir}/{trt[1]}/train/*'))
        else:
            feats = 128
        
        if acts_extracted == False:
            model = load_model(mm[1], trt[1],feats)
        
        
        for ll in layer_type:
            print(f"Extracting timeseries for {mm[1]}_{trt[1]}_{ll}...")
            global layer, sublayer
            layer = ll
            sublayer = sublayer_type

            if acts_extracted == True:
                print('Acts loaded...')
                frame_acts = np.load(f'{out_dir}/{mm[1]}_{trt[1]}_{ll}_{vid}_allframes.npy')
            else:
                frame_acts = extract_acts(model, f'{stim_dir}/{vid}')
            
                #save full model timeseries (all frames)
                np.save(f'{out_dir}/{mm[1]}_{trt[1]}_{ll}_{vid}_allframes', frame_acts)
            
            
            

            print('downsampling and running PCA...')
            #downsample to scale of fmri
            downsample_ts = down_sample(frame_acts)

            #convolve to hrf
            hrf_ts = convolve_hrf(downsample_ts)
            #pdb.set_trace()
            #standardize activations
            #hrf_ts = stats.zscore(hrf_ts)

            #hrf_ts = np.isnan(hrf_ts).any(axis=1)
            
            
            #calculate components for pca
            #n_comp = calc_pc_n(extract_pc(hrf_ts),pca_perc)
            
            #calculate final set of PCs
            #pca = extract_pc(hrf_ts, n_comp)
            #final_ts = pca.transform(hrf_ts) #reduce dimensionality of data using model PCs
            final_ts = hrf_ts
            
            
            #plot pc variance explained

            

            np.save(f'{out_dir}/{mm[1]}_{trt[1]}_{ll}_{vid}_ts', final_ts)
            
            
            #print(tc, ee,ll, n_comp)

'''
for nc, tc in enumerate(train_type):
    n_classes = len(glob(f'{train_dir}/{tc}/train/*'))

    
    model = models.__dict__[model_arch](out_feat=n_classes)
    
    for ll in layer:
        model =  model_funcs.remove_layer(model, ll)

        #extract acts for all frames of video
        frame_acts = model_loop(model, loader)
        np.save(f'fmri_data/{vid}_{model_type}_{tc}_{ee}_{ll}_allframes', frame_acts)

        #downsample to scale of fmri
        downsample_ts = down_sample(frame_acts)

        #calculate components for pca
        n_comp = calc_pc_n(extract_pc(downsample_ts),.95)
        
        #calculate final set of PCs
        pca2 = extract_pc(downsample_ts, n_comp)
        model_pcs = pca2.transform(downsample_ts) #reduce dimensionality of data using model PCs
        #plot pc variance explained

        #convolve to hrf
        final_ts = convolve_hrf(model_pcs)
        np.save(f'{out_dir}/{vid}_{model_type}_{tc}_{ee}_{ll}_TS', final_ts)
        print(tc, ee,ll, n_comp)
'''
