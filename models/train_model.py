## Train CorNet-Z on object only ImageNet
# 
# Removed all categories from ImageNet that have faces (people and non-human animals)
# 

# Load dependancies

import sys
import os, argparse
from collections import OrderedDict
import torch

import torch.nn as nn
import torchvision
from torchvision import transforms
import cornet
from torchvision import datasets
import torchvision.models as models
import numpy as np
import sys
import pdb
import model_funcs
import load_without_faces
import random
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--data', required=False,
                    help='path to folder that contains train and val folders', 
                    default=None)
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--arch', default='cornet_z',
                    help='which model to train')
parser.add_argument('--epochs', default=20, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')

global args, best_prec1
args = parser.parse_args()



writer = SummaryWriter()
print_file = False   

#Image directory

n_classes = 1200
writer = SummaryWriter(f'runs/classify_{cond}')
image_dir = f"/scratch/vayzenbe/{cond}/"
weights_dir =f"/lab_data/behrmannlab/vlad/ginn/model_weights"
model_funcs.reproducible_results(1)
start_epoch = 0
model = model_funcs.load_model('classify',cond,start_epoch, weights_dir, n_classes)

#These are all the default learning parameters from the run_CorNet script
lr = .01 #Starting learning rate
step_size = 10 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = 30
n_save = 5 #save model every X epochs


optimizer = torch.optim.SGD(model.parameters(),
                                         lr,
                                         momentum=momentum,
                                         weight_decay=weight_decay)


#lr updated given some rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
criterion = nn.CrossEntropyLoss()
criterion.cuda()

#Transformations for ImageNet
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])


train_dir = image_dir + 'train'
val_dir = image_dir + 'val'
exclude_im = f"/lab_data/behrmannlab/image_sets/imagenet_face_files.csv"
exclude_folder = f"/lab_data/behrmannlab/image_sets/imagenet_animal_classes.csv"
train_dataset = load_without_faces.load_stim(train_dir, exclude_im, exclude_folder, transform=transform)
val_dataset = load_without_faces.load_stim(val_dir, exclude_im, exclude_folder, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True)

print("data loaded")
model.cuda()
# Start training loop

valid_loss_min = np.Inf # track change in validation loss
nTrain = 1
nVal = 1
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        # move tensors to GPU if CUDA is available
        
        data, target = data.cuda(), target.cuda()
            #print('moved to cuda')
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        #print(output.shape)
        # calculate the batch loss
        loss = criterion(output, target)
        #print(loss)
        writer.add_scalar("Raw Train Loss", loss, nTrain) #write to tensorboard
        writer.flush()
        nTrain = nTrain + 1
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        
        # update training loss
        train_loss += loss.item()*data.size(0)
        #print(train_loss)
    

    scheduler.step()
    ######################    
    # validate the model #
    ######################
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in valloader:
            # move tensors to GPU if CUDA is available
            
            data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            writer.add_scalar("Raw Validation Loss", loss, nVal) #write to tensorboard
            writer.flush()
            nVal = nVal + 1
            #print('wrote to tensorboard')
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)

            topP, topClass = output.topk(1, dim=1) #get top 1 response
            equals = topClass == target.view(*topClass.shape) #check how many are right
            accuracy += torch.mean(equals.type(torch.FloatTensor)) #calculate acc; equals needed to made into a flaot first

            

    
    # calculate average losses
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(valloader.sampler)
    

    if print_file == True:
        sys.stdout = open("CorNet_Object_Out.txt", "w")

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss),
        "Test Accuracy: {:.3f}".format(accuracy/len(valloader)))
    writer.add_scalar("Average Train Loss", train_loss, epoch) #write to tensorboard
    writer.add_scalar("Average Validation Loss", valid_loss, epoch) #write to tensorboard
    writer.add_scalar("Average Acc", accuracy/len(valloader), epoch) #write to tensorboard
    writer.flush()
    
    # save model if validation loss has decreased
    if epoch % n_save == 0 or epoch == 1:
        
        file_path = f'{weights_dir}/cornet_{model_type}_{cond}_{epoch}.pt'
        model_funcs.save_model(model, epoch, optimizer, loss, scheduler,file_path)
        
        valid_loss_min = valid_loss

    if print_file == True:
        sys.stdout.close()


writer.close()


