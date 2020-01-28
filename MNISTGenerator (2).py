import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Input, Lambda, Reshape, Flatten, UpSampling2D, MaxPooling2D
from keras.models import Model
import keras.backend as K
from keras import initializers


class MNISTGenerator():

    def __init__(self):
        self.latent_dim = 50        # Dimension of Latent Representation
        self.Encoder = None
        self.Decoder = None
        self.model = None
        self.weights_path = './model weights/modified_mnist.h5'

        
    def GenerateModel(self):
        b_f = 128
        # ENCODER
        input_ = Input(shape=(56,56,1))

        encoder_hidden1 = Conv2D(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(input_)
        encoder_hidden1 = BatchNormalization()(encoder_hidden1)
        encoder_hidden1 = Activation('relu')(encoder_hidden1)

        encoder_hidden2 = Conv2D(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden1)
        encoder_hidden2 = BatchNormalization()(encoder_hidden2)
        encoder_hidden2 = Activation('relu')(encoder_hidden2)

        encoder_hidden3 = Conv2D(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(encoder_hidden2)
        encoder_hidden3 = BatchNormalization()(encoder_hidden3)
        encoder_hidden3 = Activation('relu')(encoder_hidden3)

        encoder_hidden4 = Flatten()(encoder_hidden3)

        # Latent Represenatation Distribution, P(z)
        z_mean = Dense(self.latent_dim, activation='linear', 
                                  kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden4)
        z_std_sq_log = Dense(self.latent_dim, activation='linear', 
                                  kernel_initializer= initializers.he_normal(seed=None))(encoder_hidden4)

        # Sampling z from P(z)
        def sample_z(args):
            mu, std_sq_log = args
            epsilon = K.random_normal(shape=(K.shape(mu)[0], self.latent_dim), mean=0., stddev=1.)
            z = mu + epsilon * K.sqrt( K.exp(std_sq_log)) 
            return z

        z = Lambda(sample_z)([z_mean, z_std_sq_log])


        # DECODER
        decoder_hidden0 = Dense(K.int_shape(encoder_hidden4)[1], activation='relu', kernel_initializer= initializers.he_normal(seed=None))(z)
        decoder_hidden0 = Reshape(K.int_shape(encoder_hidden3)[1:])(decoder_hidden0)

        decoder_hidden1 = Conv2DTranspose(filters = b_f*4, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden0)
        decoder_hidden1 = BatchNormalization()(decoder_hidden1)
        decoder_hidden1 = Activation('relu')(decoder_hidden1)

        decoder_hidden2 = Conv2DTranspose(filters = b_f*2, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden1)
        decoder_hidden2 = BatchNormalization()(decoder_hidden2)
        decoder_hidden2 = Activation('relu')(decoder_hidden2)

        decoder_hidden3 = Conv2DTranspose(filters = b_f, kernel_size = 2, strides = (2,2), padding = 'valid', kernel_initializer = 'he_normal' )(decoder_hidden2)
        decoder_hidden3 = BatchNormalization()(decoder_hidden3)
        decoder_hidden3 = Activation('relu')(decoder_hidden3)

        decoder_hidden4 = Conv2D(filters = 1, kernel_size= 1, strides = (1,1), padding='valid', kernel_initializer = 'he_normal')(decoder_hidden3)
        decoder_hidden4 = Activation('sigmoid')(decoder_hidden4)
        # MODEL
        vae = Model(input_, decoder_hidden4)

        # Encoder Model
        encoder = Model(inputs = input_, outputs = [z_mean, z_std_sq_log])
        
        # Decoder Model
        no_of_encoder_layers = len(encoder.layers)
        no_of_vae_layers = len(vae.layers)

        decoder_input = Input(shape=(self.latent_dim,))
        decoder_hidden = vae.layers[no_of_encoder_layers+1](decoder_input)

        for i in np.arange(no_of_encoder_layers+2 , no_of_vae_layers-1):
            decoder_hidden = vae.layers[i](decoder_hidden)
        decoder_hidden = vae.layers[no_of_vae_layers-1](decoder_hidden)
        decoder = Model(decoder_input,decoder_hidden )

        self.VAE = vae
        self.Encoder = encoder
        self.Decoder = decoder

    def LoadWeights(self):
        self.VAE.load_weights(self.weights_path)

    def GetModels(self):
        return self.VAE, self.Encoder, self.Decoder
                
