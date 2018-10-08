import sys
import os

from keras.utils import Sequence

import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical

import math
import random

from ast import literal_eval

import src.utils.utils as ut


def load_image(im):
    width = ut.params.image_width
    heigth = ut.params.image_heigth
    return img_to_array(load_img(im, grayscale=False, target_size=(heigth, width))) / 255.


class DataSequence(Sequence):
    """
    Keras Sequence object to train a model on larger-than-memory data.
    """
    def __init__(self, df, data_path, batch_size, classmode, modelmode, mode='train'):
        self.df = df
        self.bsz = ut.params.batch_size
        self.mode = mode
        self.classmode = classmode
        self.modelmode = modelmode

        # Take labels and a list of image locations in memory
        self.labels = to_categorical(np.array(self.df['category'].values.tolist()))
        self.labels_pc = to_categorical(np.array(self.df['product_category'].values.tolist()))
        self.labels_pt = to_categorical(np.array(self.df['product_type'].values.tolist()))
        self.labels_pd = to_categorical(np.array(self.df['product_details'].values.tolist()))
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
        idx_min = idx * self.bsz
        idx_max = (idx + 1) * self.bsz
        return np.array(self.labels[idx_min: idx_max])

    def get_batch_labels_pc(self, idx):
        # Fetch a batch of labels
        idx_min = idx * self.bsz
        idx_max = (idx + 1) * self.bsz
        return np.array(self.labels_pc[idx_min: idx_max])

    def get_batch_labels_pt(self, idx):
        # Fetch a batch of labels
        idx_min = idx * self.bsz
        idx_max = (idx + 1) * self.bsz
        return np.array(self.labels_pt[idx_min: idx_max])

    def get_batch_labels_pd(self, idx):
        # Fetch a batch of labels
        idx_min = idx * self.bsz
        idx_max = (idx + 1) * self.bsz
        return np.array(self.labels_pd[idx_min: idx_max])

    def get_batch_texts(self, idx):
        # Fetch a batch of labels
        idx_min = idx * self.bsz
        idx_max = (idx + 1) * self.bsz
        return np.array(self.text_list[idx_min: idx_max])

    def get_batch_features(self, idx):
        # Fetch a batch of inputs
        idx_min = idx * self.bsz
        idx_max = (idx + 1) * self.bsz
        return np.array([load_image(im) for im in
                         self.im_list[idx_min: idx_max]])

    def __getitem__(self, idx):
        if(self.classmode == 'multiclass'):
            batch_y = self.get_batch_labels(idx)
            if(self.modelmode == 'image'):
                batch_i = self.get_batch_features(idx)
                return batch_i, batch_y
            elif(self.modelmode == 'text'):
                batch_t = self.get_batch_texts(idx)
                return batch_t, batch_y
            else:
                batch_i = self.get_batch_features(idx)
                batch_t = self.get_batch_texts(idx)
                return [batch_i, batch_t], batch_y
        elif(self.classmode == 'multilabel'):
            batch_y_pc = self.get_batch_labels_pc(idx)
            batch_y_pt = self.get_batch_labels_pt(idx)
            batch_y_pd = self.get_batch_labels_pd(idx)
            if(self.modelmode == 'image'):
                batch_i = self.get_batch_features(idx)
                return batch_i, [batch_y_pc, batch_y_pt, batch_y_pd]
            elif(self.modelmode == 'text'):
                batch_t = self.get_batch_texts(idx)
                return batch_t, [batch_y_pc, batch_y_pt, batch_y_pd]
            else:
                batch_i = self.get_batch_features(idx)
                batch_t = self.get_batch_texts(idx)
                return [batch_i, batch_t], [batch_y_pc, batch_y_pt, batch_y_pd]
        else:
            print('Choose proper classmode')
            sys.exit()
