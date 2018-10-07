# split into words
import os
import sys
import pandas as pd
import numpy as np
from ast import literal_eval

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
from collections import Counter

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import src.utils.utils as ut

#nltk.download('all')


def save_df(data, safe_to):
    data.to_csv(safe_to, sep=';')
    del data


def read_df(load_from):
    df = pd.read_csv(load_from, sep=';', header=0)
    if('Unnamed: 0' in df.columns):
        df.drop(['Unnamed: 0'], axis=1)
    for col in ['reduced_title', 'tokenized']:
        if(col in df.columns):
            df.loc[:, col] = df.loc[:, col].apply(lambda x: literal_eval(x))
    return df


class preparation:
    def __init__(self, clean_title):
        self.clean_title = clean_title

    def __text_cleaning(self, text):
        """
        Tokenizing and cleaning proceedure for text data.
        """
        deadlist = ['mit', 'xxl', 'xxxl', 'uvp', 'xcm', 'grs', 'grm', 'grl',
                    'tlg', 'xxcm', 'xcm']
        transfer = {
            ord('ä'): 'ae',
            ord('ö'): 'oe',
            ord('ü'): 'ue',
            ord('ß'): 'ss'
        }
        # tokenize the text string
        tokens = word_tokenize(text)

        # convert to lower case
        tokens = [w.lower() for w in tokens]

        # transfer German umlauts into vowels
        tokens = [w.translate(transfer) for w in tokens]

        # remove punctuation and digits from each word
        table = str.maketrans('', '', string.punctuation + string.digits)
        stripped = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if word.isalpha()]

        # reduce words to their stemms
        porter = PorterStemmer()
        stemmed = list(set([porter.stem(word) for word in words]))

        # filter out
        #   stop words,
        #   words that are contained in the deadlist,
        #   words that are shorter than 3 characters and
        #   words which are assembled only from one and
        #      the same identical character
        stop_words = set(stopwords.words(['english', 'german']) + deadlist)
        words = [w for w in stemmed if w not in stop_words
                 and len(w) > 2 and len(Counter(w)) > 1]

        # et voilà
        return words

    def make_clean_sku(self, data):
        data['imagename'] = data['sku'].astype(str) + '.jpg'
        return data

    def make_clean_title(self, data):
        """
        Tokenize and stemm the title column by creating the new column
        'reduced_title'.
        For simplicity keep only those references which have a non-vanishing
        reduced_title.
        """
        if('reduced_title' not in data.columns):
            data['reduced_title'] = data['title']\
                                    .apply(lambda x: self.__text_cleaning(x))
            data = data[data['reduced_title'].apply(lambda x: len(x) > 0)]
        return data.reset_index(drop=True)

    def make_clean_imagecontent(self, data):
        """
        Keep only references to images in the DataFrame
        which exist and are not empty
        """
        indices = []
        for idx, row in data.iterrows():
            src = os.path.join(ut.dirs.original_dataset_dir,
                               str(row.sku)+'.jpg')
            if(os.path.exists(src) is False):
                indices.append(idx)
            elif(os.path.isfile(src) is False):
                indices.append(idx)
            else:
                if(os.stat(src).st_size == 0):
                    indices.append(idx)
                else:
                    pass
        return data.drop(data.index[indices]).reset_index(drop=True)

    def make_expanded_categories(self, data):
        """
        Expand the category column to enable multilabel classification
        """
        if(('category' in data.columns)):
            expanded_cats = data['category'].str.split(' > ')
            expanded_cats = expanded_cats.apply(pd.Series)\
                                         .rename({0: 'product_category',
                                                  1: 'product_type',
                                                  2: 'product_details'},
                                                 axis='columns')
            return data.join(expanded_cats)
        else:
            return data

    def make_keras_embeddings(self, data):
        """
        Create the word embeddings used for training the NLP model
        """
        delim = ','
        column = 'reduced_title'
        if(self.clean_title is not True):
            delim = ' '
            column = 'title'
        if(column in data.columns):
            #ut.params.n_words = np.max(np.array([len(x) for x in data.reduced_title]))
            #vocab = data[column].apply(pd.Series).stack().value_counts()


            tokenize = Tokenizer(num_words=ut.params.n_vocab,
                                 char_level=False,
                                 filters='0123456789!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                                 lower=True,
                                 split=delim)
            to_embed = data[column]
            tokenize.fit_on_texts(to_embed)
            embed_dict = pd.DataFrame.from_dict(tokenize.word_index, orient="index")
            embed_dict.to_csv(os.path.join(ut.dirs.raw_dir, 'embeddings.csv'))
            del embed_dict
            #description = tokenize.texts_to_matrix(data[column])

            embeddings = tokenize.texts_to_sequences(to_embed)
            embeddings = pad_sequences(embeddings, maxlen=ut.params.n_words)
            data['tokenized_title'] = embeddings.tolist()

        return data



    def make_clean(self, data):
        data = self.make_clean_title(data)
        data = self.make_clean_imagecontent(data)
        data = self.make_expanded_categories(data)
        data = self.make_keras_embeddings(data)
        data = self.make_clean_sku(data)
        data = data.dropna().reset_index(drop=True)
        return data


class stat_selection:
    def __init__(self, column='category', quant=None, sample_size=None):
        self.column = column
        self.quant = quant
        self.sample_size = sample_size

    def count_categories(self, data):
        if(self.column in data.columns):
            return pd.DataFrame(data[self.column]
                                .value_counts())\
                                .reset_index()\
                                .rename({'index': self.column,
                                         self.column: 'counts'},
                                        axis='columns')
        else:
            print(self.column, 'not in DataFrame columns!')
            print('Select from', data.columns)
            sys.exit()

    def select_category_threshold(self, data):
        if(self.quant is not None):
            df_cats = self.count_categories(data)
            cutoff = None
            if(type(self.quant) is float):
                cutoff = df_cats.counts.quantile(self.quant)
                print('Select from', self.column, 'of the DataFrame all entries which',
                      'belong to the than', self.quant, 'percentile')
            elif(type(self.quant) is int):
                cutoff = self.quant
                print('Select from', self.column, 'of the DataFrame all entries with',
                      'more than', cutoff, 'samples')
            else:
                print('Cutoff has wrong data type')
                sys.exit()
            if(cutoff is not None):
                list_cats = df_cats[df_cats['counts'] > cutoff][self.column].tolist()
                del df_cats
                return data[data[self.column].isin(list_cats)].reset_index(drop=True)
            else:
                print('Choose a different quantile.')
                sys.exit()
        else:
            return data.reset_index(drop=True)

    def select_equalized_subsample(self, data):
        if(self.sample_size is not None):
            df_cats = self.count_categories(data)
            df_sub = pd.DataFrame()
            for idx, cat_rows in df_cats.iterrows():
                if(cat_rows.counts > 100/self.sample_size):
                    cat = cat_rows[self.column]
                    df_temp = data[data[self.column] == cat]\
                                                .sample(frac=self.sample_size,
                                                        random_state=ut.params.seed)
                    df_sub = pd.concat([df_sub, df_temp])
                    del df_temp
            del df_cats
            del data
            return df_sub.reset_index(drop=True)
        else:
            return data.reset_index(drop=True)

    def make_selection(self, data):
        data = self.select_category_threshold(data)
        data = self.select_equalized_subsample(data)
        return data


def working_df(clean_title=True, column='category', quantile=None, sample_size=None):
    df_clean_dir = os.path.join(ut.dirs.raw_dir, ut.df_names.cleaned_df)

    df_return = None

    if(os.path.exists(df_clean_dir) is False or os.stat(df_clean_dir).st_size == 0):
        df_dir = os.path.join(ut.dirs.raw_dir, ut.df_names.original_df)
        df = read_df(df_dir).dropna().reset_index(drop=True)
        df_cleaned = preparation(clean_title).make_clean(df)
        save_df(df_cleaned, df_clean_dir)
        del df
    else:
        df_cleaned = read_df(df_clean_dir).dropna()


    df_return = stat_selection(column, quantile, sample_size).make_selection(df_cleaned)


    del df_cleaned
    return df_return
