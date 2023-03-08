"""
Dataloader which loads images with a folder and returns both the image and folder
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
import natsort
from PIL import Image
from glob import glob as glob

class load_stim(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        #all_imgs = os.listdir(main_dir)
        all_imgs = glob(main_dir + '/**/*.JPEG', recursive=True) + glob(main_dir + '/**/*.jpg', recursive=True)
        self.total_imgs = natsort.natsorted(all_imgs)
        


    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        
        return tensor_image, self.total_imgs[idx]
