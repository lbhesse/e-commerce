import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import sys

from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras import backend as K

import numpy as np
from ast import literal_eval

import src.utils.utils as ut
import src.utils.model_utils as mu
import src.models.model as md
import src.models.data_generator as dg

import src.data.dataframe as dat


def softmax(vector):
    ex = np.exp(vector)
    return ex/np.sum(ex)


def make_input(modelmode, img, text):
    if(modelmode == 'image'):
        return img
    elif(modelmode == 'text'):
        return text
    else:
        return [img, text]



def decode_output(classmode, df, df_row, class_prediction):

    if(classmode == 'multilabel'):
        catlist = ['cat_product_category', 'cat_product_type', 'cat_product_details']
    else:
        catlist = ['cat_category']
    for idx, cat in enumerate(catlist):
        y_true = df_row[cat]
        y_pred = (class_prediction[idx])[0]

        decoded_y_true = df_row[cat[4:]]
        decoded_y_pred = df[df[cat] == np.argmax((y_pred))][cat[4:]].unique()[0]
        print('***', cat)
        print('y_true:', decoded_y_true)
        print('y_pred:', decoded_y_pred)
        print('correctly predicted:', decoded_y_true == decoded_y_pred)
        print('')

def predict(classmode, modelmode, model=None):
    if(model is not None):
        test = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.test_df))
        if(classmode == 'multilabel'):
            catlist = ['cat_product_category', 'cat_product_type', 'cat_product_type']
        else:
            catlist = ['cat_category']
        for cat in catlist:
            cats = to_categorical(np.array(test[cat].values.tolist()))
            test['new_'+ cat] = cats.tolist()
        print(test.columns)
        tokenized_titles = test['tokenized_title'].apply(lambda x: literal_eval(x)).values.tolist()
        for idx, row in test.iterrows():
            filename = ut.dirs.test_dir + '/' + str(row.imagename)
            img = mu.load_image(filename, ut.params.image_heigth, ut.params.image_width)
            text = np.array(tokenized_titles[idx]).reshape(1, -1)

            input = make_input(modelmode, img, text)
            class_prediction = model.predict(input)
            print('*********', idx, filename)
            decode_output(classmode, test, row, class_prediction)
            print('')
            if idx > 10:
                break

    else:
        print("define pretrained model")
        sys.exit()

def predict_all(classmode, modelmode, model=None):
    test = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.test_df))
    nclasses = mu.get_n_classes(test, classmode)


    testdata = dg.DataSequence(test,
                               ut.dirs.test_dir,
                               batch_size=ut.params.batch_size,
                               classmode=classmode,
                               modelmode=modelmode)

    predictions = model.evaluate_generator(generator=testdata)

    print('Evaluate model performance on test set: <model> / <random_guess>')
    if(classmode == 'multilabel'):
        print('accuracy product_category: {0:1.2f} / {1:1.2f}'.format(predictions[4], 1./nclasses[0]))
        print('accuracy product_type:     {0:1.2f} / {1:1.2f}'.format(predictions[5], 1./nclasses[1]))
        print('accuracy product_details:  {0:1.2f} / {1:1.2f}'.format(predictions[6], 1./nclasses[2]))
        print('')
    else:
        print('category:: {0:1.2f} / {1:1.2f}'.format(predictions[1], 1./nclasses))
        print('')
    print(predictions)

@click.command()
@click.option('--classmode', type=str, default=ut.params.classmode,
                    help='choose a classmode:\n\
                            multilabel, multiclass\n\
                            (default: multilabel)')
@click.option('--modelmode', type=str, default=ut.params.modelmode,
                    help='choose a modelmode:\n\
                            image, text, combined\n\
                            (default: combined)')
def main(classmode, modelmode):
    classmode, modelmode = ut.check_modes(classmode, modelmode)
    model = mu.load_pretrained_model(classmode, modelmode)
    predict_all(classmode, modelmode, model)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
