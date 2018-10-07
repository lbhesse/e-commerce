
import os

import tensorflow as tf
from tensorflow import keras
from keras.utils import Sequence

import pandas as pd
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical

import math
import random

from ast import literal_eval

datapath = './im/'

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
        self.labels = to_categorical(np.array(self.df['category'].values.tolist()))
        self.im_list = self.df['imagename'].apply(lambda x: os.path.join(data_path, x)).tolist()
        self.text_list = self.df['tokenized_title'].apply(lambda x: literal_eval(x)).values.tolist()

    def __len__(self):
        return int(math.ceil(len(self.df) / float(self.bsz)))

    def on_epoch_end(self):
        # Shuffles indexes after each epoch if in training mode
        self.indexes = range(len(self.im_list))
        if self.mode == 'train':
            self.indexes = random.sample(self.indexes, k=len(self.indexes))

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_texts(self, idx):
        # Fetch a batch of labels
        return np.array(self.text_list[idx * self.bsz: (idx + 1) * self.bsz])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        return np.array([load_image(im) for im in self.im_list[idx * self.bsz: (1 + idx) * self.bsz]])

    def __getitem__(self, idx):
        batch_x = self.get_batch_features(idx)
        batch_t = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        #return batch_x,
        return [batch_x, batch_t], batch_y
