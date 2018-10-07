import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sys

from keras.models import load_model
import cv2
import numpy as np

import src.utils.utils as ut
import src.data.crop as cr


def load_pretrained_model(modelname):
    # load the pretrained model
    global model
    load_model_from = os.path.join(ut.dirs.model_dir, modelname + '.h5')

    model = load_model(load_model_from)
    model._make_predict_function()
    return model


def interprete_prediction(prediction):
    class_dict = {0: 'bad', 1: 'good'}
    rp = int(round(prediction[0][0]))
    return prediction[0][0], rp, class_dict.get(rp)


def predict(custom_model=None):
    if(custom_model is not None):
        for feature in ['good', 'bad']:
            for file in Path(ut.dirs.test_dir + '/' + feature).iterdir():
                if(file.name.endswith(('.jpeg'))):
                    filename = ut.dirs.test_dir + '/' + feature + '/' + file.name
                    print(filename)
                    img = cv2.imread(filename)

                    wi = ut.params.cropwidth

                    if(img.size > 3*wi**2):
                        print(img.shape)
                        img = cr.crop_image(img)
                    img = cv2.resize(img, (ut.params.image_width,
                                           ut.params.image_heigth))
                    img = np.reshape(img, [1,
                                           ut.params.image_width,
                                           ut.params.image_heigth,
                                           3])

                    classes = custom_model.predict(img)

                    print(interprete_prediction(classes))
    else:
        print("define pretrained model")
        sys.exit()

@click.command()
@click.option('--modelname', type=str, default='vgg16',
                    help='choose a model:\n\
                            vgg16:  pretrained vgg16\n\
                            cnn:    simple CNN\n\
                            (default: vgg16)')
def main(modelname):
    model = load_pretrained_model(modelname)
    predict(model)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
