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

Each category can be decomposed into a tripple of subcategories, e.g.

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
