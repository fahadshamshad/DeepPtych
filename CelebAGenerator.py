import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, BatchNormalization, Activation, Dense, Conv2DTranspose, Input, Lambda, Reshape, Flatten, UpSampling2D, MaxPooling2D
from keras.models import Model, Sequential
import keras.backend as K
from keras import initializers


class CelebAGenerator():

    def __init__(self):
        self.latent_dim = 100        # Dimension of Latent Representation
        self.GAN = None
        self.weights_path = './model weights/celeba.h5'

        
    def GenerateModel(self):
        gf_dim = 64
        gan = Sequential()
        gan.add(Dense(8192, use_bias = True, bias_initializer='zeros', input_dim=100))
        #gan.add(Reshape([-1,s16,s16,gf_dim*8]))  old one
        gan.add(Reshape([4,4,gf_dim*8]))
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) # look into scale if error and axis
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(gf_dim*4, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) # look into scale if error and axis
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(gf_dim*2, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) # look into scale if error and axis
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(gf_dim*1, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(BatchNormalization(epsilon = 1e-5,momentum = 0.9,scale = True)) # look into scale if error and axis
        gan.add(Activation('relu'))
        gan.add(Conv2DTranspose(3, 5, strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = initializers.random_normal(stddev=0.02), bias_initializer = 'zeros')) #see in channel value and std_value for random normal
        gan.add(Activation('tanh')) 

        self.GAN = gan


    def LoadWeights(self):
        self.GAN.load_weights(self.weights_path)

    def GetModels(self):
        return self.GAN

if __name__ == '__main__':
    celeba = CelebAGenerator()
    celeba.GenerateModel()
    celeba.LoadWeights()
    gan = celeba.GetModels()
    for _ in range(100):
        pred = gan.predict(np.random.randn(1,100))[0,:,:,:]
        pred = (pred+1)/2
        plt.imshow(pred)
        plt.show()    
