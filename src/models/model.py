from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dropout, Flatten, Dense, Embedding, concatenate
from keras import applications
from keras import Model

import src.utils.utils as ut


# If you want to specify input tensor
class multiclass_models:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

    def vgg16(self):
        input_tensor = Input(shape=self.input_shape)
        vgg_model = applications.VGG16(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block5_conv3'].output
        print(x.shape)

        # Stacking a new simple convolutional network on top of it
        custom_flat = Flatten()(x)
        custom_dense = Dense(256, activation='relu')(custom_flat)
        custom_drop = Dropout(0.5)(custom_dense)
        custom_out = Dense(self.n_classes, activation='sigmoid')(custom_drop)
        if(self.n_classes > 1):
            custom_out = Dense(self.n_classes,
                               activation='softmax')(custom_drop)

        # Creating new model.
        # Please note that this is NOT a Sequential() model.
        custom_model = Model(input=vgg_model.input, output=custom_out)

        # Make sure that the pre-trained bottom layers are not trainable
        for layer in vgg_model.layers:
            layer.trainable = False

        return custom_model

    def NLP(self):
        model_input  = Input(shape=(ut.params.n_words, ))
        hidden_layer = Embedding(ut.params.n_vocab, 500, input_length=ut.params.n_words)(model_input)
        hidden_layer = Flatten()(hidden_layer)
        output_layer = Dense(self.n_classes, activation='linear')(hidden_layer)


        # Creating new model.
        # Please note that this is NOT a Sequential() model.
        custom_model = Model(input=model_input, output=output_layer)

        return custom_model

    def vgg16_NLP(self):
        input_tensor = Input(shape=self.input_shape)
        vgg_model = applications.VGG16(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block5_conv3'].output
        print(x.shape)

        # Stacking a new simple convolutional network on top of it
        vgg_model_flat = Flatten()(x)
        vgg_model_dense = Dense(256, activation='relu')(vgg_model_flat)
        vgg_model_drop = Dropout(0.5)(vgg_model_dense)
        # Make sure that the pre-trained bottom layers are not trainable
        for layer in vgg_model.layers:
            layer.trainable = False

        NLP_input  = Input(shape=(ut.params.n_words, ))
        NLP_embedding = Embedding(ut.params.n_vocab, 50, input_length=ut.params.n_words)(NLP_input)
        NLP_flatten = Flatten()(NLP_embedding)


        # Let's concatenate the question vector and the image vector:
        merged = concatenate([vgg_model_drop, NLP_flatten])

        # And let's train a logistic regression over 1000 words on top:
        output = Dense(self.n_classes, activation='linear')(merged)
        # This is our final model:
        custom_model = Model(inputs=[vgg_model.input, NLP_input],
                          outputs=[output])

        return custom_model
