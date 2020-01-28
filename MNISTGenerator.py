import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Input, Lambda, Reshape, Flatten, UpSampling2D, MaxPooling2D
from keras.models import Model
import keras.backend as K
from keras import initializers


'''
Training Details:
    1. 
'''

class MNISTGenerator():

    def __init__(self):
        self.latent_dim = 20        # Dimension of Latent Representation
        self.Encoder = None
        self.Decoder = None
        self.model = None
        self.weights_path = './model weights/mnist.h5'

        
    def GenerateModel(self):
        
        input_ = Input(shape=(784,))
        # Encoder
        encoder_hidden1 = Dense(500,activation='relu', )(input_)
        encoder_hidden2 = Dense(500,activation='relu')(encoder_hidden1)

        # Latent Represenatation Distribution, P(z)
        z_mean = Dense(self.latent_dim, activation='linear')(encoder_hidden2)
        z_std_sq_log = Dense(self.latent_dim, activation='linear')(encoder_hidden2)

        # Sampling z from P(z)
        def sample_z(args):
            mu, std_sq_log = args
            epsilon = K.random_normal(shape=(K.shape(mu)[0], self.latent_dim), mean=0., stddev=1.)
            z = mu + epsilon * K.sqrt( K.exp(std_sq_log)) 
            return z
        
        z = Lambda(sample_z)([z_mean, z_std_sq_log])


        # Decoder/Generator hidden layers
        decoder_hidden1 = Dense(500, activation = 'relu',  )(z)
        decoder_hidden2 = Dense(500, activation = 'relu')(decoder_hidden1)
        ouput_ = Dense(784, activation='sigmoid')(decoder_hidden2)

        # models
        vae = Model(input_, ouput_)
        
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
                

if __name__ == "__main__":

    mnistGAN = MNISTGenerator()
    mnistGAN.GenerateModel()
    mnistGAN.LoadWeights()
    vae, encoder, decoder = mnistGAN.GetModels()
    
    z = np.random.normal(0,1,[1,latent_dim ])
    x = decoder.predict(z).reshape(28,28)
    plt.imshow(x)