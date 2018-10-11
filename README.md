e-commerce
==============================

This deep learning demo project accomplishes the task of product classification into different categories based on image and text data.  The categories serve as labels for the supervised learning approach.


The entire data set contains 99.943 images of shop items belonging to 365 different categories.  The data set is imbalanced among these categories:

![image of imbalanced data](/reports/figures/imbalanced_data.png)

To abbreviate the training time the data set is restricted as follows:
* only 10 % of the images
* only categories with at least 10 items

This abbreviated approach does not capture the whole complexity of the task but provides a first proof of concept.

The remaining data set contains then 9.171 images distributed among 141 categories and still sketches the class imbalance in a decent way:

![image of imbalanced data](/reports/figures/cleaned_imbalanced_data.png)

## Labels

Each category can be decomposed into a triple of subcategories, e.g.

    category = product category > product type > product details.

With this, one can choose between

1. multiclass classification based on the item's

        category

2. multilabel (and still mutliclass) classification based on the category decomposition into

        product category, product type, product details

The product category contains three classes distributed as

![image of imbalanced data](/reports/figures/product_categories.png)

while the product types among these three product categories are distributed as follows

![image of imbalanced data](/reports/figures/product_types.png)

One could now further break down each product type into its product details.  However, to keep it simple, the product detail distributions among each product type is not shown here.  

## Text data
The text data used for training is the product title of the web shop.  110 items are missing a product title in the original data set.  However, the subset considered during the following is assembled such that each product has a non-empty title.  

The number of words per title vary slightly among the product categories and the corresponding distributions are slightly skewed:

![image of imbalanced data](/reports/figures/title_lengths.png)

We find that the Kunst (art) product category has both the largest median and the largest average of title length, followed by Schmuck (jewelry) and finally Taschen (bags).

However, when filtering out stop words, digits, other obstacles and stemming the words contained in the titles we find a more equalized distribution of title lengths among the product categories:

![image of imbalanced data](/reports/figures/red_title_lengths.png)

These 'cleaned' titles are then used as data for the classifier.  


A more detailed investigation of the statistics of the data set can be found in the notebook

    notebooks/dataset_statistics.ipynb

# 1. Out of the box product classification

1. Clone the repository
    ```
    git clone git@github.com:L2Data/e-commerce.git
    cd e-commerce
    ```
2. Load the `sample_images` folder containing the entire images and the descriptive `.csv` file set into

        data/raw

3. Load the pre-trained models (if available) into

        models/

4. Create the subset of images with cleaned item titles by running
    ```
    make data
    ```
   Next to the original `.csv` a 'cleaned' version of the `.csv` is created.  This 'cleaned' version contains added columns such as a tokenized, e.g., cleaned title of each item, as well as the subcategories described above.  The creation of the 'cleaned' `.csv` is useful, as tokenizing the title takes some time.  When then in the further process a new subset of the entire data set is created the operations on the title have not to be done again as the 'cleaned' `.csv` will be used for all further steps once it is created.

   Furthermore, the `data` folder shows now the following structure

        .
        ├── external
        ├── interim
        ├── processed
        │   ├── test
        │   ├── train
        │   └── validate
        └── raw
            └── sample_images

    where `processed/train`, for instance, contains the training images copied from the full set of images.  

    The copying routine takes care of empty images and will exclude them from the further considerations.  

    Along with each image set (train, validate, test) comes a descriptive `.csv` as well as a comprehensive one.  

    The creation of the training data can be altered by several options such as

      * train-test-split ratio
      * sub-sample ratio
      * minimum number of items per category      

    All of these parameters carry default values stored in

        utils/utils.py

    Now, everything is ready to run the pre-trained models.

5.  Therefore, simply execute
      ```
      make model_predict
      ```
    from the root directory of the project.  

    This will execute the **default** pre-trained classifier which is a **combination of a deep-learning image classifier and a text classifier**.  
    The image classifier itself is a pre-trained vgg16 model, implemented in keras, along with a customized top to accomplish this particular classification task.  Using a pre-trained model for the image classifier is a good choice for various reasons such as the imbalance of the data set.  With the pre-trained model the feature extraction is accomplished a lot faster and easier than by training from scratch.  

    The **default** classification is **multilabel**.  

    Again, in

          utils/utils.py

    these default settings can be altered.  

    Furthermore, one can also alter two particular default settings by running
    ```
    make model_predict CLASSMODE=<classmode> MODELMODE=<modelmode>
    ```
    where `<classmode>` is  

          a. multilabel (**default**)
          b. multiclass

    and `<modelmode>`

          a. combined (**default**, run a combined image and text classifier simultaneously)
          b. image (run only the image classifier)
          c. text (run only the text classifier)

    Before altering these default settings make sure that there is the corresponding pre-trained model available in `<models>`.


With the **default** classifier trained for only 10 epochs on a NVIDIA® Tesla® V100, the model already achieves
```
Evaluate model performance on test set: <model> / <random_guess>
accuracy product_category: 0.88 / 0.33
accuracy product_type:     0.65 / 0.03
accuracy product_details:  0.42 / 0.01

average:                   0.65 / 0.12
```
Surprisingly, the text classifier in multilabel mode and trained on a CPU (Intel® Core™ i7-6500U CPU @ 2.50GHz × 4) performed also quite well  
```
Evaluate model performance on test set: <model> / <random_guess>
accuracy product_category: 0.90 / 0.33
accuracy product_type:     0.60 / 0.03
accuracy product_details:  0.32 / 0.01

average:                   0.61 / 0.12
```

However, in multiclass mode:
```
Evaluate model performance on test set: <model> / <random_guess>
category:: 0.29 / 0.01
```

# 2. Training a model
Therefore, simply execute
  ```
  make model_train
  ```
from the root directory of the project.  Make sure that it has been accounted for step 2. - 4. of the previous section such that the pre-trained model as well as the data is ready.   

If `<classmode>` and/or `<modelmode>` should be altered, either run
```
make model_train CLASSMODE=<classmode> MODELMODE=<modelmode>
```
with settings as described in the previous section, or change them in `utils/utils.py`.  There also the hyperparameters such as
  * batch batch
  * learning rate
  * epochs

as well as the image dimension can be altered.  

The training can easily be tracked by TensorBoard.  Therefore, after starting the training, open a new terminal window and run
```
tensorboard --logdir=<path-to-project>/e-commerce/logs/
```
Now, open the browser and run
```
http://localhost:6006
```

# 3. Visual search
Find images in the data set which are similar to a given image.  

This task can be accomplished by a two-fold approach:

1.  Extract the features of the images from the pre-trained model.  This yields a feature vector of floats for every image.  These feature vectors are stored in `models/extracted_features.npz`

    To extract your own features, execute
    ```
    make model_extract_features
    ```
    Make sure that the pre-trained image classifier is already in the  `models` directory.

2.  Run a simple classifier (here kNN with k=5) to find the 5 feature vector with the smallest distance to the given image.  

    To run the visual search, use the notebook

          notebooks/visual_search.ipynb

To get an impression of the functionality of this approach:
![image of imbalanced data](/reports/figures/image_search_Kunst.png)
![image of imbalanced data](/reports/figures/image_search_Taschen.png)
![image of imbalanced data](/reports/figures/image_search_Schmuck.png)



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# e-commerce
