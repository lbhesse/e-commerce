import sys
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dropout, Flatten, Dense, Embedding, concatenate, multiply
from keras import applications
from keras import Model


from keras import optimizers

import numpy as np

import src.utils.utils as ut

class basic_models:
    def __init__(self):
        pass

    def imagemodel(self):
        input_tensor = Input(shape=(ut.params.image_heigth, ut.params.image_width, 3), name='im_input')
        vgg_model = applications.VGG16(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        x = layer_dict['block5_conv3'].output

        vgg_model_flat = Flatten(name='im_flatten_1')(x)
        vgg_model_dense = Dense(128, activation='relu', name='im_dense_1')(vgg_model_flat)
        vgg_model_drop = Dropout(0.5, name='im_drop_1')(vgg_model_dense)

        custom_model = Model(input=vgg_model.input, output=vgg_model_drop)

        for layer in vgg_model.layers:
            layer.trainable = False

        return custom_model

    def textmodel(self):
        NLP_input  = Input(shape=(ut.params.n_words, ), name='text_input')
        NLP_embedding = Embedding(ut.params.n_vocab, 128, input_length=ut.params.n_words, name='text_embedding')(NLP_input)
        NLP_flatten = Flatten(name='text_flatten_1')(NLP_embedding)
        NLP_dense = Dense(128, activation='relu', name='text_dense_1')(NLP_flatten)

        custom_model = Model(input=NLP_input, output=NLP_dense)

        return custom_model


    def combined(self):
        image = self.imagemodel()
        text = self.textmodel()

        merged = multiply([image.output, text.output], name='merged')
        merged_dense = Dense(128, activation='relu', name='merged_dense_1')(merged)
        merged_drop = Dropout(0.5, name='merged_drop_1')(merged_dense)

        custom_model = Model(inputs=[image.input, text.input],
                          output=merged_drop)

        return custom_model

class custom(basic_models):
    def __init__(self, classmode, modelmode, n_classes=None):
        self.classmode = classmode
        self.modelmode = modelmode
        if(self.classmode == 'multilabel'):
            assert (len(n_classes) > 1)
            self.n_classes_pc = n_classes[0]
            self.n_classes_pt = n_classes[1]
            self.n_classes_pd = n_classes[2]
        else:
            assert (type(n_classes) != list)
            self.n_classes = n_classes

    def make_model(self):
        if(self.classmode == 'multiclass'):
            model = None
            if(self.modelmode == 'image'):
                model = basic_models.imagemodel(self)
            elif(self.modelmode == 'text'):
                model = basic_models.textmodel(self)
            else:
                model = basic_models.combined(self)

            custom_out = Dense(self.n_classes, activation='sigmoid', name='output')(model.output)
            if(self.n_classes > 1):
                custom_out = Dense(self.n_classes, activation='softmax', name='output')(model.output)
            custom_model = Model(input=model.inputs, output=[custom_out])

            return custom_model

        elif(self.classmode == 'multilabel'):
            model = None
            if(self.modelmode == 'image'):
                model = basic_models.imagemodel(self)
            elif(self.modelmode == 'text'):
                model = basic_models.textmodel(self)
            else:
                model = basic_models.combined(self)

            custom_out_pc = Dense(self.n_classes_pc, activation='sigmoid', name='output_1')(model.output)
            custom_out_pt = Dense(self.n_classes_pt, activation='sigmoid', name='output_2')(model.output)
            custom_out_pd = Dense(self.n_classes_pd, activation='sigmoid', name='output_3')(model.output)
            if(self.n_classes_pc > 1):
                custom_out_pc = Dense(self.n_classes_pc,
                                   activation='linear', name='output_1')(model.output)
            if(self.n_classes_pt > 1):
                custom_out_pt= Dense(self.n_classes_pt,
                                   activation='softmax', name='output_2')(model.output)
            if(self.n_classes_pd > 1):
                custom_out_pd = Dense(self.n_classes_pd,
                           activation='softmax', name='output_3')(model.output)
            custom_model = Model(input=model.inputs, output=[custom_out_pc, custom_out_pt, custom_out_pd])

            return custom_model
        else:
            print('Make proper classmode and modelmode choice')
            sys.exit()

    def make_compiled_model(self, learning_rate):
        if(self.classmode == 'multiclass'):
            model = self.make_model()
            model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                          loss='mse',
                          metrics=['accuracy'])
            return model
        elif(self.classmode == 'multilabel'):
            model = self.make_model()
            model.compile(optimizer=optimizers.Adam(lr=learning_rate),
                          loss=['mse', 'categorical_crossentropy', 'categorical_crossentropy'],
                          #loss_weights=[1,1,2],
                          metrics=["accuracy"])
            return model
        else:
            print('Make proper classmode and modelmode choice')
            sys.exit()
