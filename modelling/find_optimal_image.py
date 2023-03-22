"""
Find optimal image for an encoding model
"""
curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'

import sys
sys.path.insert(1, f'{curr_dir}')
sys.path.insert(1, f'{curr_dir}/fmri')
import os
import shutil

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pdb
import ginn_params as params

from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge

from joblib import dump, load

import torch

import torch.nn as nn
import torchvision
from torchvision import transforms
from torchvision import datasets
import numpy as np

import models
import pdb
import model_funcs

import random
from glob import glob as glob
import load_stim_folders as load_stim

exp = 'aeronaut'
study_dir,subj_dir, sub_list, vid, fmri_suf, fix_tr, data_dir, vols, tr, fps, bin_size, ages = params.load_params(exp)
#find the optimal layer/PCs for region using full dataset
# 
#loop through layers
# 
model_arch = 'cornet_z_sl'
train_types = ['imagenet_noface', 'imagenet_oneface', 'imagenet_vggface', 'vggface_oneobject', 'vggface', 'random']
out_feats = [600,601,1200,601,600,600]
layers = ['V1','V2','V4','pIT','aIT', 'decoder']
sub_layer = ['output','output','output','output','output','avgpool']

seed = 1

model_dir = '/lab_data/behrmannlab/vlad/ginn/modelling'
weights_dir = f'{model_dir}/model_weights'
encode_dir = f'{model_dir}/encoding_models'
results_dir = f'{curr_dir}/results/top_ims'
file_suf = '_face'

encoding_summary = pd.read_csv(f'{encode_dir}/{model_arch}_encoding_summary{file_suf}.csv')

summary_type = 'model'
rois = ['FFA_face','EVC_face','A1_face']

#load data from arguments
train_type = sys.argv[1]
roi = sys.argv[2]
age = sys.argv[3]
n_feats = out_feats[train_types.index(train_type)]

print(f'loading data for {train_type} {roi} {age}')

n_images = 10 #images to keep

##CHANGE ME TO TRAIN!!!
im_dir = '/lab_data/behrmannlab/image_sets/imagenet_vggface/train'

transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_best_layer(train_type, roi, age):
    #loop through layers and select the best one
    #extract layer from encoding summary 
    roi = roi.split('_')[0]
    
    model_df = encoding_summary[(encoding_summary['train_type'] == train_type) & (encoding_summary['roi'] == roi) & (encoding_summary['age'] == age)]

    #find layer with best score
    best_layer = model_df.loc[model_df['score'].idxmax()]['layer']

    #pull optimal pc
    n_pcs = model_df.loc[model_df['score'].idxmax()]['optimal_pc']


    

    return best_layer, n_pcs

    

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


def load_encoding_model(model_type,layer, roi,age):
    #Load encoding model
    encoding_model = load(f'{encode_dir}/{model_arch}_{model_type}_{layer}_{roi}_{age}_ridge.joblib')

    #load PCA transformation
    pca = load(f'{encode_dir}/{model_arch}_{model_type}_{layer}_{roi}_{age}_pca.joblib')


    return encoding_model, pca
    

#set up hook to specified layer
def _store_feats(layer, inp, output):
    """An ugly but effective way of accessing intermediate model features
    """
    output = output.cpu().numpy()
    
    _model_feats.append(np.reshape(output, (len(output), -1)))


'''
Start of main script
'''

#find best layer
layer, n_pcs = find_best_layer(train_type, roi, age)
sublayer = sub_layer[layers.index(layer)]


#load model
model = load_model(model_arch, train_type,n_feats=n_feats)

#load encoding model
encoding_model, pca = load_encoding_model(train_type,layer, roi,age)



try:
    m = model.module
except:
    m = model
model_layer = getattr(getattr(m, layer), sublayer)
model_layer.register_forward_hook(_store_feats)



im_dataset = load_stim.load_stim(im_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(im_dataset, batch_size=128, shuffle=False, num_workers = 4, pin_memory=True)
nn = 0
image_df = pd.DataFrame(columns = ['model_arch','train_type','layer','image','pred'])
print('Searching for optimal images...')
with torch.no_grad():
    for data, label in dataloader:
        
        # move tensors to GPU if CUDA is available
        data = data.cuda()
        
        #extract model features
        _model_feats = []
        model(data)
        #output = model(data)
        
        out = np.vstack(_model_feats)
        #standardize out
        out = stats.zscore(out, axis=1)

        #transform features
        pc_out = pca.transform(out)
        pc_out = pc_out[:,:n_pcs]

        #standardize
        pc_out = stats.zscore(pc_out, axis=1)

        #predict
        pred = encoding_model.predict(pc_out)

        #append images and predictions

        image_df = image_df.append(pd.DataFrame({'model_arch': model_arch, 'train_type': train_type, 'layer': layer, 'image':label, 'pred':pred}), ignore_index=True)
        #convert pred col to float
        image_df['pred'] = image_df['pred'].astype(float)

        #sort by prediction
        image_df = image_df.nlargest(n_images, 'pred')
        
        

print('saving and copying images')

#save image df
image_df.to_csv(f'{results_dir}/{model_arch}_{train_type}_{layer}_{roi}_{age}_images.csv')

#create directory to save images
if not os.path.exists(f'{results_dir}/{model_arch}_{train_type}_{layer}_{roi}_{age}'):
    os.makedirs(f'{results_dir}/{model_arch}_{train_type}_{layer}_{roi}_{age}')

#loop through images and  copy to new directory
for im in image_df['image']:
    im_file = im.split('/')[-1]
    shutil.copy(im, f'{results_dir}/{model_arch}_{train_type}_{layer}_{roi}_{age}/{im_file}')


        



