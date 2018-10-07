# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import os
import shutil
import sys

import pandas as pd
import numpy as np
import math
import random

from sklearn.model_selection import train_test_split

import src.utils.utils as ut
import src.data.dataframe as dat


def cleanup():
    for dirs in [ut.dirs.train_dir, ut.dirs.validation_dir, ut.dirs.test_dir]:
        if os.path.exists(dirs):
            shutil.rmtree(dirs)

    csv_list = os.listdir(ut.dirs.processed_dir)
    if len(csv_list) != 0:
        for item in csv_list:
            if item.endswith(".csv"):
                os.remove(os.path.join(ut.dirs.processed_dir, item))


class prepare_datasets:
    def __init__(self, split, seed):
        self.split = split
        self.seed = seed

    def create_folders(self):
        dirs = []

        processed_dir = ut.dirs.processed_dir
        train_dir = ut.dirs.train_dir
        validation_dir = ut.dirs.validation_dir
        test_dir = ut.dirs.test_dir

        dirs.append(processed_dir)
        dirs.append(train_dir)
        dirs.append(validation_dir)
        dirs.append(test_dir)

        for directory in dirs:
            if not os.path.exists(directory):
                os.mkdir(directory)

    def make_split(self, data):
        if(data is not None and len(data) != 0):
            train_tmp, test = train_test_split(data,
                                               test_size=self.split,
                                               random_state=self.seed)
            split_tmp = len(test)/len(train_tmp)
            train, valid = train_test_split(train_tmp,
                                            test_size=split_tmp,
                                            random_state=self.seed)

            assert len(valid) == len(test)
            assert len(test) + len(valid) + len(train) == len(data)

            dat.save_df(train, os.path.join(ut.dirs.processed_dir,
                                      ut.df_names.train_df))
            dat.save_df(valid, os.path.join(ut.dirs.processed_dir,
                                    ut.df_names.valid_df))
            dat.save_df(test, os.path.join(ut.dirs.processed_dir,
                                  ut.df_names.test_df))
            del data
            return (train.reset_index(drop=True),
                    valid.reset_index(drop=True),
                    test.reset_index(drop=True))
        else:
            print("data set is not specified")
            print("stop execution")
            sys.exit()

    def get_filenames(self, data):
        filenames = []

        for idx, data_rows in data.iterrows():
            filename = str(data_rows.sku) + '.jpg'
            filenames.append(filename)

        return filenames

    def copy(self, filenames, src_dir=ut.dirs.original_dataset_dir,
             dst_dir=ut.dirs.train_dir):
        counter = 0
        corrupted = 0
        for name in filenames:
            src = os.path.join(src_dir, name)
            dst = os.path.join(dst_dir, name)
            if(os.path.exists(src) and (os.stat(src).st_size != 0)):
                if(os.path.exists(dst)):
                    pass
                else:
                    shutil.copy(src, dst)
                    counter += 1
            elif(os.path.exists(src) and (os.stat(src).st_size == 0)):
                print('...  found corrupted image', os.path.exists(src), src)
                corrupted += 1
        if(counter != 0):
            print('... copied', counter, 'files from', src_dir, 'to', dst_dir)
            if(corrupted != 0):
                print('... thereby found', corrupted, 'files')
        if(counter == 0 and corrupted == 0):
            print('... datasets already created earlier')

    def make_datasets(self, data):
        self.create_folders()
        train, valid, test = self.make_split(data)

        sample_names = []
        for samples in [train, valid, test]:
            sample_names.append(self.get_filenames(samples))
            del samples

        original_dir = ut.dirs.original_dataset_dir

        sample_dirs = [ut.dirs.train_dir,
                       ut.dirs.validation_dir,
                       ut.dirs.test_dir]

        for names, dirs in zip(sample_names, sample_dirs):
            self.copy(names, original_dir, dirs)










@click.command()
@click.option('--split', type=float, default=0.2,
                    help='validation and test split (default: 0.2)')
@click.option('--seed', type=int, default=42,
                    help='random seed (default: 42)')
@click.option('--clean', type=int, default=1,
                    help='delete training, validation and test data\n\
                     before creating new\n\
                     0: False, 1: True (default: 1)')
@click.option('--clean_title', type=int, default=1,
                     help='delete training, validation and test data\n\
                      before creating new\n\
                      0: False, 1: True (default: 1)')
@click.option('--quantile', default=.5,
                    help='crop training (train, valid, test) \
                     data to show only the targets\n\
                     None: False, (float, int): True (default: 0.5)')
@click.option('--subsample_size', default=.1,
                    help='crop training (train, valid, test) \
                     data to show only the targets\n\
                     None: False, (float, int): True (default: 0.5)')
def main(split, seed, clean, clean_title, quantile, subsample_size):
    category = 'category'
    cl_title = True
    if(clean_title == 0):
        print('clean training data')
        cl_title = False

    df = dat.working_df(cl_title, category, quantile, subsample_size)
    if(clean == 1):
        print('clean training data')
        cleanup()

    print('creating training data')
    prepare_datasets(split, seed).make_datasets(df)

    print('\nDone')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
