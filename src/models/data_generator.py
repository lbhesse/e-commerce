
import os

import tensorflow as tf
from tensorflow import keras
from keras.utils import Sequence

import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical

import math
import random

from ast import literal_eval

datapath = './im/'

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

def load_image(im):
    return img_to_array(load_img(im, grayscale=False, target_size=(64, 64))) / 255.




class DataSequence(Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, data_path, batch_size, mode='train'):
        self.df = df
        self.bsz = batch_size
        self.mode = mode

        # Take labels and a list of image locations in memory
        self.labels1 = to_categorical(np.array(self.df['product_category'].values.tolist()))
        self.labels2 = to_categorical(np.array(self.df['product_type'].values.tolist()))
        self.labels3 = to_categorical(np.array(self.df['product_details'].values.tolist()))
        self.im_list = self.df['imagename'].apply(lambda x: os.path.join(data_path, x)).tolist()
        self.text_list = self.df['tokenized_title'].apply(lambda x: literal_eval(x)).values.tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels1(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels1[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_labels2(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels2[idx * self.bsz: (idx + 1) * self.bsz])
        
    def get_batch_labels3(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels3[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_texts(self, idx):
        # Fetch a batch of labels
        return np.array(self.text_list[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([load_image(im) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_t = self.get_batch_texts(idx)
        batch_y1 = self.get_batch_labels1(idx)
        batch_y2 = self.get_batch_labels2(idx)
        batch_y3 = self.get_batch_labels3(idx)
        #return batch_x,
        return [batch_x, batch_t], [batch_y1, batch_y2, batch_y3]
