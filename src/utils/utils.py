import os
import sys

import settings

global rootdir
rootdir = settings.PROJECT_ROOT

def check_modes(classmode, modelmode):
    cm = None
    mm = None
    if(len(classmode) == 0):
        cm = params.classmode
        print('No classmode chosen.  Set to default:', classmode)
    elif(classmode not in ['multilabel', 'multiclass']):
            print('choose from available classmodes:')
            print('   multiclass, multilabel')
            sys.exit()
    else:
        print('********************', classmode)
        cm = classmode

    if(len(modelmode) == 0):
        mm = params.modelmode
        print('No modelmode chosen.  Set to default:', modelmode)
    elif(modelmode not in ['combined', 'image', 'text']):
        print('chose from available modelmodes:')
        print('   combined, image, text')
        sys.exit()
    else:
        mm = modelmode
    return cm, mm

class dirs:
    base_dir = os.path.join(rootdir, "data")
    raw_dir = os.path.join(base_dir, "raw")
    original_dataset_dir = os.path.join(raw_dir, "sample_images")
    processed_dir = os.path.join(base_dir, "processed")
    train_dir = os.path.join(processed_dir, "train")
    validation_dir = os.path.join(processed_dir, "validate")
    test_dir = os.path.join(processed_dir, "test")
    model_dir = os.path.join(rootdir, "models")

class df_names:
    original_df = "201802_sample_product_data.csv"
    cleaned_df = "201802_cleaned_product_data.csv"
    train_df = "201802_train_product_data.csv"
    valid_df = "201802_valid_product_data.csv"
    test_df = "201802_test_product_data.csv"
    extracted_features = 'extracted_features.npz'

class params:
    n_words = 9
    n_vocab = 5000    #8214
    seed = 42
    quantile = 10
    subsample = .10
    batch_size = 8
    epochs = 12
    learning_rate = 0.00005
    image_width = 64
    image_heigth = 64
    classmode = 'multilabel'
    modelmode = 'combined'


def main():
    pass


if __name__ == '__main__':
    main()
