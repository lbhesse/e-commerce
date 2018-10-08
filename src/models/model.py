from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dropout, Flatten, Dense, Embedding, concatenate, multiply
from keras import applications
from keras import Model

import src.utils.utils as ut


# If you want to specify input tensor
class multiclass_models:
    def __init__(self, input_shape, n_classes1, n_classes2, n_classes3):
        self.input_shape = input_shape
        self.n_classes1 = n_classes1
        self.n_classes2 = n_classes2
        self.n_classes3 = n_classes3

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
        output1 = Dense(self.n_classes1, activation='sigmoid')(custom_drop)
        output2 = Dense(self.n_classes2, activation='sigmoid')(custom_drop)
        output3 = Dense(self.n_classes3, activation='sigmoid')(custom_drop)
        if(self.n_classes > 1):
            custom_out1 = Dense(self.n_classes1,
                               activation='softmax')(custom_drop)
            custom_out2= Dense(self.n_classes2,
                               activation='softmax')(custom_drop)
            custom_out3 = Dense(self.n_classes3,
                       activation='softmax')(custom_drop)

        # Creating new model.
        # Please note that this is NOT a Sequential() model.
        custom_model = Model(input=vgg_model.input, output=[output1, output2, output3])

        # Make sure that the pre-trained bottom layers are not trainable
        for layer in vgg_model.layers:
            layer.trainable = False

        return custom_model

    def NLP(self):
        model_input  = Input(shape=(ut.params.n_words, ))
        hidden_layer = Embedding(ut.params.n_vocab, 128, input_length=ut.params.n_words)(model_input)
        hidden_layer = Flatten()(hidden_layer)
        #output_layer = Dense(self.n_classes, activation='linear')(hidden_layer)

        output1 = Dense(self.n_classes1, activation='sigmoid')(hidden_layer)
        output2 = Dense(self.n_classes2, activation='sigmoid')(hidden_layer)
        output3 = Dense(self.n_classes3, activation='sigmoid')(hidden_layer)
        if(self.n_classes1 > 1):
            custom_out1 = Dense(self.n_classes1,
                               activation='softmax')(hidden_layer)
            custom_out2= Dense(self.n_classes2,
                               activation='softmax')(hidden_layer)
            custom_out3 = Dense(self.n_classes3,
                       activation='softmax')(hidden_layer)
        # Creating new model.
        # Please note that this is NOT a Sequential() model.
        custom_model = Model(input=model_input, output=[output1, output2, output3])

        return custom_model

    def vgg16_NLP(self):
        input_tensor = Input(shape=self.input_shape)
        vgg_model = applications.VGG16(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        # Creating dictionary that maps layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
        for layer in vgg_model.layers:
            print(layer.name)
        # Getting output tensor of the last VGG layer that we want to include
        x = layer_dict['block5_conv3'].output#conv_pw_13_bn'].output
        print(x.shape)

        # Stacking a new simple convolutional network on top of it
        vgg_model_flat = Flatten()(x)
        vgg_model_dense = Dense(128, activation='relu')(vgg_model_flat)
        vgg_model_drop = Dropout(0.5)(vgg_model_dense)
        # Make sure that the pre-trained bottom layers are not trainable
        for layer in vgg_model.layers:
            layer.trainable = False

        NLP_input = Input(shape=(ut.params.n_words, ))
        NLP_embedding = Embedding(ut.params.n_vocab, 128, input_length=ut.params.n_words)(NLP_input)
        NLP_flatten = Flatten()(NLP_embedding)
        NLP_dense = Dense(128, activation='relu')(NLP_flatten)


        # Let's concatenate the question vector and the image vector:
        merged = multiply([vgg_model_drop, NLP_dense])
        merged = Dense(128, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        # And let's train a logistic regression over 1000 words on top:
        output1 = Dense(self.n_classes1, activation='sigmoid')(merged)
        output2 = Dense(self.n_classes2, activation='sigmoid')(merged)
        output3 = Dense(self.n_classes3, activation='sigmoid')(merged)
        if(self.n_classes1 > 1):
            custom_out1 = Dense(self.n_classes1,
                               activation='softmax')(merged)
            custom_out2= Dense(self.n_classes2,
                               activation='softmax')(merged)
            custom_out3 = Dense(self.n_classes3,
                       activation='softmax')(merged)
        # This is our final model:
        custom_model = Model(inputs=[vgg_model.input, NLP_input],
                          outputs=[output1, output2, output3])

        return custom_model
