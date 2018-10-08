import os
import sys
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import tensorflow as tf


from ast import literal_eval

import src.utils.utils as ut
import src.models.model as md
import src.models.data_generator as dg

import src.data.dataframe as dat

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k,
                    v in logs.items() if k.startswith('val_')}

        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

def make_labels(data, classmode):
    #print(len(data['cat_product_category'].value_counts()))
    n_classes = len(data['cat_category'].value_counts())
    n_classes1 = len(data['cat_product_category'].value_counts())
    n_classes2 = len(data['cat_product_type'].value_counts())
    n_classes3 = len(data['cat_product_details'].value_counts())

    if(classmode == 'multiclass'):
        return n_classes
    else:
        return [n_classes1, n_classes2, n_classes3]

def train(classmode, modelmode, batch_size, epochs, learning_rate):
    train = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.train_df))
    nclasses = make_labels(train, classmode)

    valid = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.valid_df))
    make_labels(valid, classmode)

    traindata = dg.DataSequence(train,
                                ut.dirs.train_dir,
                                batch_size=batch_size,
                                classmode=classmode,
                                modelmode=modelmode)
    validdata = dg.DataSequence(valid,
                                ut.dirs.validation_dir,
                                batch_size=batch_size,
                                classmode=classmode,
                                modelmode=modelmode)

    model = md.custom(classmode, modelmode, nclasses).make_compiled_model(learning_rate)
    model.summary()

    save_model_to = os.path.join(ut.dirs.model_dir, classmode + '_' + modelmode + '.h5')

    Checkpoint = ModelCheckpoint(save_model_to,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    Earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0,
                              mode='auto',
                              baseline=None)
    model.fit_generator(generator=traindata,
                        steps_per_epoch=len(train)//batch_size,
                        validation_data=validdata,
                        validation_steps=len(valid)//batch_size,
                        epochs=epochs,
                        callbacks=[TrainValTensorBoard(write_graph=False),
                                   Checkpoint],
                        #verbose=1,
                        use_multiprocessing=False,
                        workers=1)


@click.command()
@click.option('--classmode', type=str, default=ut.params.classmode,
                    help='choose a classmode:\n\
                            multilabel, multiclass\n\
                            (default: multilabel)')
@click.option('--modelmode', type=str, default=ut.params.modelmode,
                    help='choose a modelmode:\n\
                            image, text, combined\n\
                            (default: combined)')
@click.option('--ep', type=float, default=ut.params.epochs,
                    help='number of epochs (default: {})'.
                    format(ut.params.epochs))
@click.option('--lr', type=float, default=ut.params.learning_rate,
                    help='learning rate (default: {})'.
                    format(ut.params.learning_rate))
@click.option('--bs', type=int, default=ut.params.batch_size,
                    help='batch size (default: {})'.
                    format(ut.params.batch_size))
def main(classmode, modelmode, bs, ep, lr):
    print('************* classmode', len(classmode))
    if(len(classmode)==0):
        classmode = ut.params.classmode
        print('No classmode chosen.  Set to default:', classmode)
    if(len(modelmode)==0):
        modelmode = ut.params.modelmode
        print('No modelmode chosen.  Set to defualt:', modelmode)

    train(classmode, modelmode, bs, ep, lr)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
