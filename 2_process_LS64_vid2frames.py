#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 00:23:13 2022
https://www.geeksforgeeks.org/how-to-get-properties-of-python-cv2-videocapture-object/
https://superuser.com/questions/326629/how-can-i-make-ffmpeg-be-quieter-less-verbose

@author: mohamed
"""

import numpy as np
import os,shutil
import subprocess
import scipy
from tqdm import tqdm

# import cv2
import matplotlib.pyplot as plt
# subprocess.call('ren *.txt *.bat', shell=True)
final_data=np.zeros(shape=(64,40,40,2048))

import cv2

# video = cv2.VideoCapture("001_001_001.mp4");
# fps = video.get(cv2.CAP_PROP_FPS)
# print(fps)
# print(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Select the desired number of frames : Total_Frames
# Get Number of frames : Desired_Frames
# Step = Total_Frames // Desired_Frames

PathToLs64="/media/mohamed/Data_Linux/1_Datasets/SIGN_LANGUAGE/lsa64_cut"
LSSA64_dir=PathToLs64+"/LSA64_CLASSES/"

categories = os.listdir(LSSA64_dir)
categories.sort()
fps=30

for idx,c in enumerate(tqdm(categories[26:27])):
    print(c)
    video_files=os.listdir(LSSA64_dir+"/"+c+"/")
    Frames=PathToLs64+"/Frames/"+c
    print(Frames)
    for ind,vid in enumerate(video_files):
        vid_file=LSSA64_dir+"/"+c+"/"+vid
        Frames1=Frames+"/"+vid[0:-4]
        os.makedirs(Frames1,exist_ok=True)
        cmd=f"ffmpeg  -hide_banner -loglevel error -i {vid_file} -vf fps={fps} {Frames1}/wk%02d.jpg"
        os.system(cmd)
        


# # for idx,c in enumerate(categories):
# #     video_files=os.listdir(c+"/")
# #     for ind,vid in enumerate(video_files):
# #         arr=np.zeros(shape=(40,299,299,3))
# #         vid_file=c+"//"+vid
# #         os.mkdir('frames')
# #         cmd=f"ffmpeg  -i {vid_file} -r 10 frames/wk%02d.jpg"
# #         os.system(cmd)
#         # for i,img in enumerate(os.listdir("frames/")):
#         #     if i <22:
#         #         print(img)
#         #         img=cv2.imread("frames/"+img)
#         #         img=cv2.resize(img,(299,299))
#         #         arr[i]=img
#                 #print(str(i)+"th image")
#         # shutil.rmtree('frames')
#         # pred=model_fin.predict(arr)
#         # final_data[idx,ind]=pred
#     #print(str(ind)+"th video done")