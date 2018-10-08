import os

import settings

global rootdir
rootdir = settings.PROJECT_ROOT


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

class params:
    n_words = 9
    n_vocab = 5000
    seed = 42
    quantile = 100
    subsample = .1
    batch_size = 20
    epochs = 5
    learning_rate = 0.001
    image_width = 64
    image_heigth = 64


def main():
    pass


if __name__ == '__main__':
    main()
