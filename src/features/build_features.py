import os
import numpy as np
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np

import scipy.sparse as sp


from keras import backend as K

import src.utils.utils as ut
import src.utils.model_utils as mu
import src.data.dataframe as dat
import src.utils.sparse_matrix as sm


def feature_extract(filename, model, outputlayer=21):
    _convout1_f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[outputlayer].output])

    img = mu.load_image(filename, ut.params.image_heigth, ut.params.image_width)

    features=_convout1_f([img, 1])[0]
    features = np.squeeze(features.ravel())
    return features

def make_all_feature_extract(model):
    df = dat.read_df(os.path.join(ut.dirs.processed_dir,
                                  ut.df_names.cleaned_df))
    n_files = len(df)

    preds = sp.lil_matrix((n_files, 128))#8192))

    for idx, rows in df.iterrows():
        filename = rows['imagename']
        feature = feature_extract(filename, model)
        preds[idx] = feature
    return preds

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
    if(len(classmode)==0):
        classmode = ut.params.classmode
        print('No classmode chosen.  Set to default:', classmode)
    if(len(modelmode)==0):
        modelmode = ut.params.modelmode
        print('No modelmode chosen.  Set to defualt:', modelmode)

    print('\nStart feature extraction')

    model = mu.load_pretrained_model(classmode, modelmode)
    extracted_features = make_all_feature_extract(model)
    save_sparse_to = os.path.join(ut.dirs.model_dir, ut.df_names.extracted_features)
    sm.save_sparse_matrix(save_sparse_to, extracted_features)

    print('Finished feature extraction')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
