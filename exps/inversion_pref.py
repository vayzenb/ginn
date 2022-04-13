"""
Extracts activations from penultimate layer from object, face, and random models
to human and inverted human faces

Created on Thu Feb  6 14:02:03 2020

@author: vayze
"""


import os
#os.chdir('C:/Users/vayze/Desktop/GitHub Repos/GiNN/')

import sys
sys.path.insert(1, '/user_data/vayzenbe/GitHub_Repos/ginn/model_training')

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
from sklearn.utils import resample


import deepdish as dd
import pdb
import warnings
warnings.filterwarnings("ignore")


curr_dir = '/user_data/vayzenbe/GitHub_Repos/ginn'
#test_dir = "/user_data/vayzenbe/image_sets/"
test_dir = f"{curr_dir}/stim/"
act_dir = '/lab_data/behrmannlab/vlad/ginn/activations'
#test_dir = "/lab_data/plautlab/imagesets/"
weights_dir = '/lab_data/behrmannlab/vlad/ginn/model_weights'

train_type = ['random','imagenet_noface', 'imagenet_oneface',
'imagenet_vggface', 'vggface_oneobject', 'vggface']
train_type = ['vggface']
model_arch = ['cornet_z']
test_stim = ['cropped_face', 'schematic']

exp = 'inversion_pref'

os.makedirs(f'{act_dir}/{exp}', exist_ok = True)
os.makedirs(f"{curr_dir}/results/{exp}/", exist_ok= True)

seed = 1
alpha = .05

test_only = False

test_cond =['upright','inverted']

layer = 'decoder'
sublayer = 'linear'
suf = '_unsupervised'
iter = 10

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
    print('extracting features...', cond)
    

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


def measure_pref(upright_acts, inverted_acts, model_info):
    print('Analyzing pref...')
    
    upright_acts[upright_acts <=0] = 0
    inverted_acts[inverted_acts <=0] = 0

    pdb.set_trace()
    upright_mean = np.mean(upright_acts[upright_acts >0])
    inverted_mean = np.mean(inverted_acts[inverted_acts >0])
    #extract mean energy
    mean_pref = upright_mean/ (upright_mean + inverted_mean)

    boot_vals= []
    for ii in range(0,iter):
        #sample image with replacement
        temp_upright = resample(upright_acts, replace = True, random_state = ii)
        temp_inverted = resample(inverted_acts, replace = True, random_state = ii)

        upright_mean = np.mean(temp_upright[temp_upright >0])
        inverted_mean = np.mean(temp_inverted[temp_inverted >0])

        temp_pref = upright_mean/ (upright_mean + inverted_mean)
        boot_vals.append(temp_pref)

    ci_low = np.percentile(boot_vals, alpha*100)
    ci_high= np.percentile(boot_vals, 100-alpha*100)
    print( model_info + [mean_pref, ci_high, ci_low])
    return mean_pref, ci_high, ci_low



for mm in enumerate(model_arch):   
    for trt in enumerate(train_type):
        model = load_model(mm[1], trt[1])

        summary_df = pd.DataFrame(columns = ['model_arch', 'train_type', 'test_stim','pref','ci_low','ci_high'])
        for ts in test_stim:
            test_ims = f'{test_dir}/{ts}'
            file_name = f'{mm[1]}_{trt[1]}_{ts}'

            if test_only == True:
                upright_acts = dd.io.load(f"{act_dir}/{exp}/{file_name}_upright{suf}.h5")
                inverted_acts = dd.io.load(f"{act_dir}/{exp}/{file_name}_inverted{suf}.h5")
            else:
                upright_acts = extract_acts(model, test_ims, 'upright')
                inverted_acts = extract_acts(model, test_ims, 'inverted')

                
                dd.io.save(f"{act_dir}/{exp}/{file_name}_upright{suf}.h5", upright_acts)
                dd.io.save(f"{act_dir}/{exp}/{file_name}_inverted{suf}.h5", inverted_acts)

                
                model_info = [mm[1], trt[1], ts]
                pref, ci_low, ci_high = measure_pref(upright_acts,inverted_acts, model_info)
                summary_df = summary_df.append(pd.Series(model_info +[pref, ci_low, ci_high],index = summary_df.columns),ignore_index = True)
                summary_df.to_csv(f"{curr_dir}/results/{exp}/{file_name}.csv", index=False)
                
                print(mm[1], ts, 'Saved')


