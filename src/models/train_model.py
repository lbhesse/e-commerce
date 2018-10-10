import os
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from keras.callbacks import ModelCheckpoint, EarlyStopping

import src.utils.utils as ut
import src.utils.model_utils as mu
import src.models.model as md
import src.models.data_generator as dg

import src.data.dataframe as dat


def train(classmode, modelmode, batch_size, epochs, learning_rate):
    train = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.train_df))
    nclasses = mu.ref_n_classes(classmode)

    valid = dat.read_df(os.path.join(ut.dirs.processed_dir, ut.df_names.valid_df))

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
                        callbacks=[mu.TrainValTensorBoard(write_graph=False),
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

    project_dir = Path(__file__).resolve().parents[2]

    load_dotenv(find_dotenv())

    main()
