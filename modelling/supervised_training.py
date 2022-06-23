## Train CorNet-Z on object only ImageNet
# 
# Removed all categories from ImageNet that have faces (people and non-human animals)
# 

# Load dependancies

import sys

import os, argparse, shutil
from collections import OrderedDict
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
from torch.utils.tensorboard import SummaryWriter

print('libs loaded')

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--data', required=False,
                    help='path to folder that contains train and val folders', 
                    default=None)
parser.add_argument('-o', '--output_path', default=None,
                    help='path for storing ')
parser.add_argument('--arch', default='cornet_z',
                    help='which model to train')
parser.add_argument('--epochs', default=30, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--rand_seed', default=1, type=int,
                    help='Seed to use for reproducible results')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

suf=''

image_dir= '/scratch/vayzenbe/'
#image_dir ='/lab_data/behrmannlab/image_sets/'
out_dir = '/lab_data/behrmannlab/vlad/ginn/model_weights'

print('read some args')

global args, best_prec1
args = parser.parse_args()

model_funcs.reproducible_results(args.rand_seed)

#create tensorboard summary writer
image_type = args.data
image_type=image_type.replace(image_dir, '')
model_type = f'{args.arch}_{image_type[:-1]}{suf}'
print(model_type)
writer = SummaryWriter(f'runs/{model_type}')
best_prec1 = 0

print_file = False   

#Image directory

n_classes = len(glob(f'{args.data}/train/*'))

start_epoch = 1

#These are all the default learning parameters from the run_CorNet script
lr = .01 #Starting learning rate
step_size = 10 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = args.epochs
n_save = 5 #save model every X epochs

def save_checkpoint(state, is_best, epoch, filename='checkpoint.pth.tar'):
    torch.save(state, f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar')
    if (epoch) == 1 or (epoch) % n_save  == 0:
        shutil.copyfile(f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar', f'{out_dir}/{filename}_{epoch}_{args.rand_seed}.pth.tar')
    if is_best:
        shutil.copyfile(f'{out_dir}/{filename}_checkpoint_{args.rand_seed}.pth.tar', f'{out_dir}/{filename}_best_{args.rand_seed}.pth.tar')



model = models.__dict__[args.arch](out_feat=n_classes)
model.cuda()
print('model created..')


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
# optionally resume from a checkpoint
if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))


train_dir = args.data + 'train'
val_dir = args.data + 'val'
train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True)

print("data loaded")

# Start training loop

print('starting training...')
valid_loss_min = np.Inf # track change in validation loss
nTrain = 1
nVal = 1
for epoch in range(start_epoch, n_epochs+1):

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
        writer.add_scalar("Supervised Raw Train Loss", loss, nTrain) #write to tensorboard
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
            writer.add_scalar("Supervised Raw Validation Loss", loss, nVal) #write to tensorboard
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

    prec1 = accuracy/len(valloader)
    is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss),
        "Test Accuracy: {:.3f}".format(accuracy/len(valloader)))
    writer.add_scalar("Supervised Average Train Loss", train_loss, epoch) #write to tensorboard
    writer.add_scalar("Supervised Average Validation Loss", valid_loss, epoch) #write to tensorboard
    writer.add_scalar("Supervised Average Acc", accuracy/len(valloader), epoch) #write to tensorboard
    writer.flush()
    
    # save model if validation loss has decreased
    save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best,epoch,filename=f'{model_type}')


writer.close()


