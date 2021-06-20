import torch.nn as nn
import torch
import cornet
import collections

def make_classify_model(model,n_classes):
    """
    Change decoder layer into classifier with specified number of classes
    
    Inputs are:
    model
    n_classes
    """
    decode_layer = list(model[5][:])
    del decode_layer[2] #remove original linear layer
    decode_layer.insert(2, nn.Linear(1024, n_classes))
    decode_layer = nn.Sequential(*decode_layer) #make Sequential

    model = nn.Sequential(*list(model[:-1])) #remove original decoding layer
    model.add_module('5', decode_layer) #replace with new classification  layer

    return model

def make_ae_model(model):
    """
    Change decoder layer into autoencoder
    
    Inputs are:
    model
    """
    
    convT2d = nn.ConvTranspose2d(1024, 3, 224)
    torch.nn.init.kaiming_uniform_(convT2d.weight, a=0, mode='fan_in', nonlinearity='relu') 
    decode_layer = nn.Sequential(*list(model[5][:-3]),nn.ReLU(), convT2d) #recreate decoding layer as autoencoder
    model = nn.Sequential(*list(model[:-1])) #remove original decoding layer
    model.add_module('5', decode_layer) #add AE decoding layer

    return model

def make_contrast_model(load_checkpoint, n_classes):
    """
    Change decoder layer into contrastive learning model with i have no clue
    
    Inputs are:
    no idea
    """

    return 

def make_hebbian_model(model, n_classes):
    """
    convert into hebbian learning by removing classification layer. FC layer stays.
    
    Inputs are:
    model
    """
    
    decode_layer = nn.Sequential(*list(model[5][:-2]), nn.Linear(1024, n_classes)) #recreate decoding layer as autoencoder
    model = nn.Sequential(*list(model[:-1])) #remove original decoding layer
    model.add_module('5', decode_layer) #reattach decoding layer

    return model

def make_ae_decoder(model):
    """
    Seperate model into seperate encoder and decoder for habituation/dishabituation. 
    
    Inputs are:
    model
    """
    #Load CorNet_Z
    model = getattr(cornet, 'cornet_z')
    model = model(pretrained=False, map_location='cpu') #load model into CPU so we can mess with it
    model = model.module  # remove DataParallel
    
    convT2d = nn.ConvTranspose2d(1024, 3, 224)
    torch.nn.init.kaiming_uniform_(convT2d.weight, a=0, mode='fan_in', nonlinearity='relu') 
    decoder = nn.Sequential(nn.ReLU(), convT2d) #recreate decoding layer as autoencoder; currently ontop of FC layer
    last_layer = nn.Sequential(*list(model[5][:-1])) #create stripped version of last decoding layer
    encoder = nn.Sequential(*list(model[:-1])) #remove original decoding layer
    encoder.add_module('5', last_layer) #replace last decoding layer with stripped version
    
    encoder = nn.DataParallel(encoder.cuda()) #move back to GPU with dataparallel
    decoder = nn.DataParallel(decoder.cuda()) #move back to GPU with dataparallel
    
    return encoder, decoder



def rename_keys(checkpoint):
    """
    Because models were trained in different ways, sometimes its necessary to rename the keys from default cornet convention
    """
    
    new_check = collections.OrderedDict()
    newkeys=[0,0,1,1,2,2,3,3,4,4,5,5]
    oldkeys = ['V1', 'V1', 'V2','V2', 'V4','V4','pIT', 'pIT','aIT','aIT','decoder', 'decoder']
    n=0
    for n,k in enumerate(checkpoint.keys()):
        new_name = k


        if oldkeys[n] == 'decoder':
            new_name = new_name.replace(oldkeys[n], f'{newkeys[n]}')
            new_name = new_name.replace('linear', '2')
        else:
            new_name = new_name.replace(oldkeys[n], f'{newkeys[n]}')

        new_check[new_name] = checkpoint[k]
        
    return new_check

def remove_layer(model, layer):
    
    lower_layers =['null','aIT','pIT', 'V4', 'V2', 'V1']
    model = model.module  # remove DataParallel
    
    if layer == "out":
        pass
    elif layer == 'avgpool':
        decode_layer = nn.Sequential(*list(model[5][0:2])) #recreate decoding layer as autoencoder
        model = nn.Sequential(*list(model[:-1])) #remove original decoding layer
        model.add_module('5', decode_layer) #add decoding layer with jsut avgpool
        
    else:
        ind = lower_layers.index(layer)
        model = nn.Sequential(*list(model[:-ind])) #remove original decoding layer
        
    model = nn.DataParallel(model.cuda()) #move back to GPU with dataparallel
    
    return model

def load_model(model_type, train_cond, start_epoch=0, weights_dir=None, n_classes=600):
    """
    model loading function that converts cornet_z into a classificaiton, autoencoder, or contrastive learning models
    
    Inputs are:
    model_type - classify, ae, contrast
    train_cond - imageset that the model was trained on
    start_epoch - set to 0 if training from scratch, else load from specified epoch
    weights_dir - directory of the weights
    n_classes - how many classes to include for classification
    """
    
    #Load CorNet_Z
    model = getattr(cornet, 'cornet_z')
    model = model(pretrained=False, map_location='cpu') #load model into CPU so we can mess with it
    model = model.module  # remove DataParallel

    if model_type == 'classify':
        model = make_classify_model(model, n_classes)
    elif model_type == 'ae':
        model = make_ae_model(model)
    elif model_type == 'contrast':
        model = make_contrast_model(model, start_epoch)
    elif model_type == 'hebbian':
        model = make_hebbian_model(model, n_classes)
    elif model_type == 'avgpool':
        model = remove_decoder(model)

    model = nn.DataParallel(model.cuda()) #move back to GPU with dataparallel
    if start_epoch > 0:
        checkpoint = torch.load(f'{weights_dir}/cornet_{model_type}_{train_cond}_{start_epoch}.pt')
        #this is set up in multiple ways because i was not consistent in how i initially trained some of these models
        model.load_state_dict(checkpoint['model_state_dict'])
        '''
        try:
            try:
                new_check = rename_keys(checkpoint['model_state_dict'])
                model.load_state_dict(new_check)
            except:
                model.load_state_dict(checkpoint['model_state_dict'])
        except:
            try:
                checkpoint = rename_keys(checkpoint)
                model.load_state_dict(checkpoint)
            except:
                model.load_state_dict(checkpoint)'''           

    return model

    
def save_model(model, epoch, optimizer, loss, scheduler, file_path):

    print('Saving model ...')
    #torch.save(model.state_dict(), f'{weights_dir}/cornet_classify_{cond}_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'scheduler_state_dict': scheduler.state_dict()
        }, file_path)


def extract_acts(model, im):
    """
    Extracts the activations for a series of images
    """
    model.eval()

    with torch.no_grad():

        im = im.cuda()
        output = model(im)
        output =output.view(output.size(0), -1)


    return output

