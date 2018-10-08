import os
import sys
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
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


def train(modelname,
          batch_size,
          epochs,
          learning_rate,
          augment,
          image_width,
          image_heigth):
    input_shape = (image_width, image_heigth, 3)


    train = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.train_df))
    train['category'] = train['category'].astype('category').cat.codes
    train['product_category'] = train['product_category'].astype('category').cat.codes
    train['product_type'] = train['product_type'].astype('category').cat.codes
    train['product_details'] = train['product_details'].astype('category').cat.codes
    #pd.DataFrame({col: df[col].astype('category').cat.codes for col in df}, index=df.index)
    valid = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.valid_df))
    valid['category'] = valid['category'].astype('category').cat.codes
    valid['product_category'] = valid['product_category'].astype('category').cat.codes
    valid['product_type'] = valid['product_type'].astype('category').cat.codes
    valid['product_details'] = valid['product_details'].astype('category').cat.codes
    #pd.DataFrame({col: df[col].astype('category').cat.codes for col in df}, index=df.index)

    n_classes = np.max(np.unique(train['category'].tolist()))+1
    n_classes1 = np.max(np.unique(train['product_category'].tolist()))+1
    n_classes2 = np.max(np.unique(train['product_type'].tolist()))+1
    n_classes3 = np.max(np.unique(train['product_details'].tolist()))+1

    seq = dg.DataSequence(train, ut.dirs.train_dir,  batch_size=50)
    vaseq = dg.DataSequence(valid, ut.dirs.validation_dir,  batch_size=50)

    mod = md.multiclass_models(input_shape, n_classes1, n_classes2, n_classes3)



    model = mod.vgg16_NLP()
    model.compile(
                  optimizer=optimizers.Adam(lr=0.03),
                  loss="mse",
                  metrics=["accuracy"])

    model.summary()

    save_model_to = os.path.join(ut.dirs.model_dir, modelname + '.h5')

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
    model.fit_generator(generator=seq,
                        steps_per_epoch=len(train)//50,
                        validation_data=vaseq,
                        validation_steps=len(valid)//50,
                        epochs=50,
                        callbacks=[
                                    TrainValTensorBoard(write_graph=False),
                                    Checkpoint#,
                                    #Earlystop
                                    ],
                        verbose=1,
                        use_multiprocessing=False,
                        workers=1)




@click.command()
@click.option('--modelname', type=str, default='vgg16',
                    help='choose a model:\n\
                            vgg16:  pretrained vgg16\n\
                            cnn:    simple CNN\n\
                            (default: vgg16)')
@click.option('--ep', type=float, default=ut.params.epochs,
                    help='number of epochs (default: {})'.
                    format(ut.params.epochs))
@click.option('--lr', type=float, default=ut.params.learning_rate,
                    help='learning rate (default: {})'.
                    format(ut.params.learning_rate))
@click.option('--augment', type=int, default=1,
                    help='data augmentation\n\
                    0: False, 1: True (default: 1)')
@click.option('--bs', type=int, default=ut.params.batch_size,
                    help='batch size (default: {})'.
                    format(ut.params.batch_size))
@click.option('--width', type=int, default=ut.params.image_width,
                    help='width of the sample images (default: {})'.
                    format(ut.params.image_width))
@click.option('--heigth', type=int, default=ut.params.image_heigth,
                    help='heigth of the sample images (default: {})'.
                    format(ut.params.image_heigth))
def main(modelname, bs, ep, lr, augment, width, heigth):

    augmentation = True
    if(augment == 0):
        augmentation = False

    train(modelname, bs, ep, lr, augmentation, width, heigth)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
