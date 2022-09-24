#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 00:09:24 2022

@author: mohamed
"""

import numpy as np
import os,shutil
import subprocess
import scipy
import cv2
import matplotlib.pyplot as plt
# subprocess.call('ren *.txt *.bat', shell=True)
final_data=np.zeros(shape=(64,40,40,2048))

from keras.models import Model

from keras.models import load_model
model_fin= load_model('modelinception7.h5', compile=False)


PATH_LSA64_FRAMES="/media/mohamed/Data_Linux/1_Datasets/SIGN_LANGUAGE/lsa64_cut/Frames/"
categories = os.listdir(PATH_LSA64_FRAMES)

def load_vid(path_to_vid):
    arr=np.zeros(shape=(40,299,299,3))
    for i,img in enumerate(os.listdir(path_to_vid)):
        # print(img)
        img=cv2.imread(path_to_vid+"/"+img)
        img=cv2.resize(img,(299,299))
        arr[i]=img
        # print(str(i)+"th image")
    return arr

# final_data=np.zeros(shape=(64,40,40,2048))
path_to_vid="/media/mohamed/Data_Linux/1_Datasets/SIGN_LANGUAGE/lsa64_cut/Frames/005_Bright/005_001_005"
arr=load_vid(path_to_vid)
print("Array shape:", arr.shape)
pred=model_fin.predict(arr)
print(pred)
# finale_data = final_data.reshape(-1,40,2048)

# for idx,c in enumerate(categories):
#     video_files=os.listdir(c+"/")
#     for ind,vid in enumerate(video_files):
#         vid_file=c+"//"+vid

#         # os.mkdir('frames')
#         # get_ipython().system('ffmpeg  -i $vid_file -r 10 frames/wk%02d.jpg')

#         # for i,img in enumerate(os.listdir("frames/")):
#         #     if i <22:
#         #         print(img)
#         #         img=cv2.imread("frames/"+img)
#         #         img=cv2.resize(img,(299,299))
#         #         arr[i]=img
#                 #print(str(i)+"th image")
#         # shutil.rmtree('frames')

#         pred=model_fin.predict(arr)
#         del arr
#         final_data[idx,ind]=pred
        
#     print(str(ind)+"the video done")

# print(final_data.shape)





