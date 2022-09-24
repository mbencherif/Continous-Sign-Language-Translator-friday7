#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:19:25 2022
https://symbiosisacademy.org/tutorial-index/pandas-read_csv-header/#:~:text=Pandas%20read_csv%20%28%29%20function%20automatically%20parses%20the%20header,this%20default%20behavior%20to%20customize%20the%20column%20names.

@author: mohamed
"""
import pandas as pd
import os
import glob2
import numpy as np


PathToLS64="/media/mohamed/Data_Linux/1_Datasets/SIGN_LANGUAGE/lsa64_cut"
PathToCsv=PathToLS64+"/LSA64-Classes.csv"
df=pd.read_csv(PathToCsv, header = 0, sep = ',')

ListClasses=list(df["Name"])
for i, cl in enumerate(ListClasses):
    os.makedirs(PathToLS64+"/LSA64_CLASSES/"+f"{i+1:03d}_"+cl, exist_ok=True)

ListVideos=glob2.glob(PathToLS64+"/all_cut/*.mp4")
ListVideos.sort()

for a, k in enumerate(ListVideos):
    ClassVideo=k.split("/")[-1].split("_")[0]
    # if ClassVideo=="052":
        # print(ClassVideo)
    cl_new=ListClasses[int(ClassVideo)-1]
    Video_Output_Dir=f"{int(ClassVideo):03d}_"+cl_new

    FullDir=PathToLS64+"/LSA64_CLASSES/"+Video_Output_Dir
    cmd="cp "+k+" "+FullDir+"/"
    print(cmd)
    os.system(cmd)
    

# for index, row in df.iterrows():
#     print(index)
#     print(df.at[row,"" ])
    
# fid=open(PathToCsv,"r")
# lines=fid.read().splitlines()
# fid.close()


