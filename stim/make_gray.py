from PIL import Image, ImageOps
from glob import glob
import pandas as pd
import pdb
import numpy as np
stim_folder = 'exp_stim'
out_folder = 'gray_exp_stim'

im_files = glob(f'{stim_folder}/*.jpg')
im_buffer = 10
#def make_gray():
for imfile in im_files:
    #imfile = f'{stim_folder}/AU02_35.jpg'
    og_im = Image.open(imfile)
    
    
    im = Image.new(og_im.mode, [og_im.size[0] + im_buffer+30, og_im.size[1] +im_buffer+30],(128, 128, 128))
    

    #t_loc = int((im_size - crop_im.size[0])/2)
    #l_loc = int((im_size- crop_im.size[1])/2)
    im.paste(og_im, (im_buffer+2, im_buffer+2)) 
    
    np_im = np.array(im)

    #pdb.set_trace()
    #Find where the red is by indexing the non-gray colors in a channel
    locs = np.argwhere(np_im[:,:,1] != 128)
    
    #find left, top, right, and bottom
    top = np.min(locs[:,0])
    left = np.min(locs[:,1])
    bottom = np.max(locs[:,0])
    right = np.max(locs[:,1])
    
    
    #Crop IM
    crop_im = im.crop((left-im_buffer, top-im_buffer,  right+im_buffer, bottom+im_buffer))
    #pdb.set_trace()
    
    #determine size of new images
    im_size = np.max(crop_im.size)
    
    #Make a new image as a square
    new_im = Image.new(crop_im.mode, [im_size, im_size],(128, 128, 128))
    
    #Determine where to paste it
    t_loc = int((im_size - crop_im.size[0])/2)
    l_loc = int((im_size- crop_im.size[1])/2)
    
    #paste into new_im
    new_im.paste(crop_im, (t_loc, l_loc)) 
    im_name = imfile.split('/')
    #new_im.save(f'{out_folder}/{im_name[1]}')
   # pdb.set_trace()
    gray_image = ImageOps.grayscale(new_im)
    #
    gray_image.save(f'{out_folder}/{im_name[1]}')


def creat_triallist():
    trial_list =pd.DataFrame(columns = ['stim1', 'stim2'])
    for sn1, s1 in enumerate(im_files):
        stim1 = s1.split('/')
        stim1 = stim1[1]
        for s2 in im_files[sn1+1:]:
            stim2 = s2.split('/')
            stim2 = stim2[1]
            #print(stim1,stim2)
            trial_list = trial_list.append(pd.Series([stim1, stim2],index = trial_list.columns), ignore_index=True)
            trial_list.to_csv('exp_trials.csv', index = False)

#pdb.set_trace()

#make_gray()




