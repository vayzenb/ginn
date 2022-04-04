import os
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class load_stim(Dataset):
    def __init__(self, main_dir,exclude_im=None, exclude_folder=None, transform=None):
        """
        A custom dataset that excludes specific images and classes. 
        Was designed to exclude any image with a human face or any classes with animal faces

        Input:
        main_dir - the image directory to load
        exclude_im - a csv file with images to exclude
        exclude_folder - a csv files with folders to exclude
        transform - pytorch transforms

        """

        self.main_dir = main_dir
        self.transform = transform
        all_classes = os.listdir(main_dir)

        
        #load files to exclude
        
        exclude_im = pd.read_csv(exclude_im)
        exclude_im = set(exclude_im['url'])
    
        #if exclude_folder != None:
            #load list of class folders to exclude
        exclude_folder = pd.read_csv(exclude_folder)    

        

        imgs =[]
        class_label =[]
        ii = 0
        for im_folder in all_classes:
            #if im_folder in exclude_folder.unique(): #check if the current folder should be exlcuded
            #    continue
            #else: #else extract images from it
            curr_ims = [os.path.join(im_folder,file) for file in os.listdir(os.path.join(main_dir, im_folder))]
            curr_ims = set(curr_ims)
            final_ims = list(curr_ims - exclude_im)
            
            imgs += final_ims
            class_label += [ii] *len(final_ims)
            ii += 1

        self.final_imgs = imgs
        self.samples = class_label
            
    def __len__(self):
        return len(self.final_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.final_imgs[idx])
        #print(img_loc)
        
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        
        return tensor_image, self.samples[idx]
