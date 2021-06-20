## Converts CorNet-Z into an autoencoder to set intial weights
# These are weights that may have arose over the course of evolution
# Removed all categories from ImageNet that have faces (people and non-human animals)
# 

# Load dependancies


import sys

import torch
torch.manual_seed(1)
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import numpy as np
#from load_stim import load_stim
import sys
import cornet
import model_funcs
from torch.utils.tensorboard import SummaryWriter
import load_without_faces

print_file = False   

print('loaded libraries')

#Image directory
model_type = 'ae'
cond = 'imagenet_objects'
writer = SummaryWriter(f'runs/ae_{cond}')
image_dir = f"/scratch/vayzenbe/{cond}/"
weights_dir =f"/lab_data/behrmannlab/vlad/ginn/model_weights"

train_resume = False
#These are all the default learning parameters from the run_CorNet script
lr = 1e-3 #Starting learning rate
step_size = 10 #How often (epochs)the learning rate should decrease by a factor of 10
weight_decay = 1e-4
momentum = .9
n_epochs = 30
n_save = 5 #save model every X epochs
start_epoch = 0

model = model_funcs.load_model('ae',cond,start_epoch, weights_dir, 0)


optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad = True,weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)


#lr updated given some rule
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size)
criterion =  nn.MSELoss()
criterion.cuda()

#Transformations for ImageNet
transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

#Load training and validation datasets
train_dir = image_dir + 'train'
val_dir = image_dir + 'val'
exclude_im = f"/lab_data/behrmannlab/image_sets/imagenet_face_files.csv"
exclude_folder = f"/lab_data/behrmannlab/image_sets/imagenet_animal_classes.csv"
train_dataset = load_without_faces.load_stim(train_dir, exclude_im, exclude_folder, transform=transform)
val_dataset = load_without_faces.load_stim(val_dir, exclude_im, exclude_folder, transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True,drop_last= True)
valloader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers = 4, pin_memory=True,drop_last= True)

print("data loaded")


# Start training loop

valid_loss_min = np.Inf # track change in validation loss
nTrain = 1
nVal = 1
for epoch in range(1, n_epochs+1):
    print('Starting training')

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    
    
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in trainloader:
        # move tensors to GPU if CUDA is available
        
        data = data.cuda()
            #print('moved to cuda')
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        #print(output.shape, data.shape)
        # calculate the batch loss
        loss = criterion(output, data)
        
        
        writer.add_scalar("Raw Train Loss", loss, nTrain) #write to tensorboard
        writer.flush()
        nTrain = nTrain + 1
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        #print(loss, nTrain)

        #clip to prevent exploding gradients
        #nn.utils.clip_grad_norm_(model.parameters,max_norm=2.0, norm_type=2)
        
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
            
            data = data.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, data)
            writer.add_scalar("Raw Validation Loss", loss, nVal) #write to tensorboard
            writer.flush()
            nVal = nVal + 1
            #print('wrote to tensorboard')
            # update average validation loss 
            valid_loss += loss.item()*data.size(0)



    #save_recon(data[0],output[0], cond, epoch)
    # calculate average losses
    train_loss = train_loss/len(trainloader.sampler)
    valid_loss = valid_loss/len(valloader.sampler)
    
    

    if print_file == True:
        sys.stdout = open("CorNet_Face_Out.txt", "w")

    # print training/validation statistics 
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    writer.add_scalar("Average Train Loss", train_loss, epoch) #write to tensorboard
    writer.add_scalar("Average Validation Loss", valid_loss, epoch) #write to tensorboard
    writer.flush()
    
    # save model if validation loss has decreased
    if epoch % n_save == 0 or epoch == 1:
        file_path = f'{weights_dir}/cornet_{model_type}_{cond}_{epoch}.pt'
        model_funcs.save_model(model, epoch, optimizer, loss, scheduler,file_path)
        print('Saving model ...')
        valid_loss_min = valid_loss

    if print_file == True:
        sys.stdout.close()


writer.close()


