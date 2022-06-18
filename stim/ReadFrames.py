# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:04:18 2019

@author: VAYZENB
"""

import os
import cv2
from PIL import Image
import pdb

#os.chdir("C:/Users/vayzenb\Desktop\GitHub Repos\LiMA")
vid_folder = "fmri_videos/"

vids = os.listdir("pixar/")


for ii in range(0,len(vids)):
    print(vids[ii])
    vidcap = cv2.VideoCapture(vid_folder + vids[ii])
    vidFile = vids[ii][:-4]
    os.makedirs(vid_folder + "frames/"+ vidFile)
    success,image = vidcap.read()
    count = 1
    success = True
    while success:    
      cv2.imwrite(vid_folder + "frames/" + vidFile + "/" + vidFile + "_" + str(count) + ".jpg", image)     # save frame as JPEG file
      success,image = vidcap.read()
      count += 1
      
    #frames = os.listdir("frames/" + vidFile + "/")

    #for ii in range(0,len(frames)):
    #    IM = Image.open("Frames/" + vidFile + "/" + frames[ii]).convert("RGB")
    #    IM = IM.crop((370, 100, 1070, 800)) 
    #    IM = IM.resize((350, 350))
    #    IM.save("Frames/" + vidFile + "/"  + frames[ii])
    
      
