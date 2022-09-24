#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:25:01 2022

@author: mohamed
"""

from keras.layers import Flatten
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K
from keras.models import Model
from keras.layers import GlobalMaxPooling2D, Dense
# K.tensorflow_backend._get_available_gpus()
from keras.preprocessing.image import ImageDataGenerator
PathToFrames="/media/mohamed/Data_Linux/1_Datasets/SIGN_LANGUAGE/lsa64_cut/Frames"

## Load Pre-Trained Model
model_sq= InceptionV3(weights='imagenet', include_top=False,input_shape=(299,299,3))
model_sq.summary()

t=GlobalMaxPooling2D()(model_sq.output)
o=Dense(64,activation = 'softmax')(t)
#Input is same as Model_sq input and output is o
model_int = Model([model_sq.input],o)
model_int.summary()
#Compiling inception
model_int.compile(optimizer = 'Adam', 
                  loss      = 'categorical_crossentropy', 
                  metrics   = ['accuracy'])


batch_size = 16
val_frac =0.1
train_datagen = ImageDataGenerator(validation_split=val_frac)

train_generator = train_datagen.flow_from_directory(
        PathToFrames,  # this is the target directory
        target_size =(299, 299),  # all images will be resized to 299x299
        batch_size  =batch_size,
        class_mode  ='categorical', 
        subset="training")

val_generator = train_datagen.flow_from_directory(
        PathToFrames,  # this is the target directory
        target_size =(299, 299),  # all images will be resized to 299x299
        batch_size  =batch_size,
        class_mode  ='categorical',
        subset      ="validation")

# csvlogger
import keras
from keras.callbacks import CSVLogger
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

csv_logger = CSVLogger('training1.log')
time_callback = TimeHistory()
model_int.fit(train_generator,
              epochs           =100,
              steps_per_epoch  =250,
              callbacks        =[csv_logger,
                                 time_callback],
              validation_data  =val_generator, 
              validation_steps =250)

times = time_callback.times

model_int1 = model_int#intact for softmax approach
model_int1.layers.pop()
fin_o      = model_int1.layers[-1].output
## Correction here inputs instead of input and outputs instead of output
model_fin  = Model(inputs  = model_int1.input, 
                   outputs = [fin_o])
model_fin.save('modelinception7.h5')


