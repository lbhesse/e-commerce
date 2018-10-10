import os
import numpy as np

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

import src.utils.utils as ut
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

def ref_n_classes(classmode):
    data = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.cleaned_df))
    n_classes = len(data['category'].value_counts())
    n_classes1 = len(data['product_category'].value_counts())
    n_classes2 = len(data['product_type'].value_counts())
    n_classes3 = len(data['product_details'].value_counts())

    if(classmode == 'multiclass'):
        print(type([n_classes, n_classes]))
        return n_classes
    else:
        return [n_classes1, n_classes2, n_classes3]


def get_n_classes(data, classmode):
    n_classes = len(data['cat_category'].value_counts())
    n_classes1 = len(data['cat_product_category'].value_counts())
    n_classes2 = len(data['cat_product_type'].value_counts())
    n_classes3 = len(data['cat_product_details'].value_counts())

    if(classmode == 'multiclass'):
        return n_classes
    else:
        return [n_classes1, n_classes2, n_classes3]


def load_pretrained_model(classmode, modelmode):
    # load the pretrained model
    global model
    load_model_from = os.path.join(ut.dirs.model_dir, classmode + '_' + modelmode + '.h5')

    model = load_model(load_model_from)
    model._make_predict_function()
    return model


def load_image(filename, heigth, width):
    img = img_to_array(load_img(filename, grayscale=False, target_size=(heigth, width))) / 255.
    return np.array(img).reshape(1, heigth, width, 3)


def load_image_for_batch(im):
    width = ut.params.image_width
    heigth = ut.params.image_heigth
    return img_to_array(load_img(im, grayscale=False, target_size=(heigth, width))) / 255.
