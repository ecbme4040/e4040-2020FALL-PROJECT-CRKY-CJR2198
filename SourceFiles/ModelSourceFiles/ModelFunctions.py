from tensorflow.keras import layers
import tensorflow as tf

## This file contains functions used to create the DCGAN generator and discriminator ##


## Convolution block function
## Used for discriminator convolutional blocks
## Takes convolution paramters and initializer for Gamma and Weights
## x_in is previous layer output
def convolution(x_in, kernal_dim, num_kernals, batch_norm, strides, activation, kernel_initializer, gamma_init):
    ## Add convolution
    x_out = layers.Conv2D(filters=num_kernals, kernel_size=kernal_dim, strides=strides,
                                   kernel_initializer=kernel_initializer,
                                   use_bias=False,
                                   padding='same')(x_in)
    # No bias due to batch norm

    ## Add batch norm
    if batch_norm == True:
        x_out = layers.BatchNormalization(gamma_initializer=gamma_init) (x_out)

    ## Apply chosen activation
    x_out = apply_activation(x_out, activation)

    ## Return output
    return x_out


## Deconvolution block function
## Used for generator deconvolutional blocks
## Takes conv2dtranspose parameters and initializer for Gamma and the Weights
## x_in is previous layer output
def deconvolution(x_in, kernal_dim, num_kernals, batch_norm, strides, activation, kernel_initializer, gamma_init):

    ## Apply conv2d transpose to input, no bias due to batch norm
    x_out = layers.Conv2DTranspose(filters=num_kernals, kernel_size=kernal_dim, strides=strides,
                                   kernel_initializer=kernel_initializer,
                                   use_bias=False,
                                   padding='same')(x_in)

    ## Apply batch norm if specified
    if batch_norm == True:
        x_out = layers.BatchNormalization(gamma_initializer=gamma_init) (x_out)

    ## Apply activation function
    x_out = apply_activation(x_out, activation)

    ## Return output
    return x_out



## Function takes output from layer and applies chosen activation according to string
def apply_activation(x_in, activation):

    if activation == 'selu':
        x_out = layers.Activation("selu")(x_in) # For HDCGAN
    if activation == "relu":
        x_out = layers.ReLU()(x_in)
    if activation == "leakyrelu":
        x_out = layers.LeakyReLU(0.2)(x_in) # Alpha = 0.2 for the Discriminator per the paper
    if activation == "tanh":
        x_out = layers.Activation("tanh")(x_in)

    return x_out

## Function takes dictionary with Generator parameters, creates and returns Keras Model instance based on specifications
## Input is dictionary
def create_generator_dict(gen_dict):

    ## Example Input Dictionary ##
    # gen_dict = {'num_latent': 128,
    #             "filter_num": [128, 64],
    #             "filter_size": [5, 5],
    #             "strides": [2, 2],
    #             'dense': (8, 8, 256),
    #             'activation': 'leakyrelu',
    #             'out_channels': 1,
    #             'latent_distribution': 'normal'}

    ## Initializers for weights and gamma in batch norm
    ## Both initialized to Random Normal
    weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)

    num_latent = gen_dict['num_latent'] # Number of latent variables to use
    dense_units = gen_dict['dense'] # number of dense units, in tuple of format (filter size, filter size, number of filters) total = filter_size * filter_size * number_of_filters
    activation = gen_dict['activation'] # Activation function to apply for all layers except output
    filters = gen_dict['filter_size'] # List with filter size dimension for each layer
    filter_num = gen_dict['filter_num'] # List with number of filters for each layer
    strides = gen_dict['strides'] # List with number of strides for each layer

    latent_input = layers.Input(shape=(num_latent,)) #Input is the number of latent variables

    ## First deconv block with project and reshape
    x = layers.Dense(dense_units[0]*dense_units[1]*dense_units[2], use_bias=False)(latent_input) # Project and reshape the latent input to the first layer
    x = layers.BatchNormalization(gamma_initializer=gamma_init)(x) # Apply batch norm
    x = apply_activation(x, activation)
    x = layers.Reshape((dense_units[0], dense_units[1], dense_units[2]))(x)


    ## Apply repeating blocks for each of the layers specified in the dictionary
    ## Iterates over the list of filters and applies a deconv block for each provided filter
    for idx in range(len(filters)):
        x = deconvolution(x_in=x, kernal_dim=filters[idx], num_kernals=filter_num[idx], batch_norm=True, strides=strides[idx], activation=activation, kernel_initializer=weight_init, gamma_init=gamma_init)

    ## Final output draws the image, stride 1 and activation tanh
    x = deconvolution(x_in=x, kernal_dim=5, num_kernals=gen_dict['out_channels'], batch_norm=False, strides=1, activation='tanh', kernel_initializer=weight_init, gamma_init=gamma_init) #no batch norm in the final layer

    model = tf.keras.Model(latent_input, x)

    ## Return constructed model
    return model


## Function takes dictionary with Discriminator parameters, creates and returns Keras Model instance based on specifications
## Input is dictionary and output shape of the generator
def create_discriminator_dict(disc_dict, input_dim):

    ## All kernels and gamma are initialized to random normals
    weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)

    activation = disc_dict['activation'] # Activation function to use at ever level
    filters = disc_dict['filter_size'] # Filter size to use at each level
    filter_num = disc_dict['filter_num'] # Number of filters to use at each level
    dropout = disc_dict['dropout'] # To apply dropout to the discriminator
    noise = disc_dict['random_noise'] # To add random noise

    x_in = layers.Input(shape=[input_dim[1], input_dim[2], input_dim[3]]) #Input based on output of generator

    # For each specified layer, apply a convolution block of conv + batch norm + activation
    for idx in range(len(filters)):
        if idx == 0:
            x = convolution(x_in, filters[idx], filter_num[idx], False, 2, activation, weight_init, gamma_init) # Don't apply batch norm at the first layer per the original paper
        else:
            x = convolution(x, filters[idx], filter_num[idx], True, 2, activation, weight_init, gamma_init)

    x = layers.Flatten()(x) # Flatten features

    if dropout > 0: # To apply dropout
     x = layers.Dropout(dropout)(x)

    if noise == True: # To apply noise
        x = layers.GaussianNoise(0.02)(x)

    x = layers.Dense(1)(x) #Output logit, not applying activation
    model = tf.keras.Model(x_in, x)

    return model

##Custom linear layer
class Linear(tf.keras.layers.Layer):
    def __init__(self, output_size=1024*4*4):
        super(Linear, self).__init__()
        self.out_size = output_size

    def build(self, input_shape):
        matrix_init = tf.random_normal_initializer(mean=0.0, stddev=0.02, seed=None)
        self.w = tf.Variable(initial_value=matrix_init(shape=(input_shape[-1], self.out_size), dtype='float32'), trainable=True) #No bias

    def call(self, inputs):
        return tf.matmul(inputs, self.w)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'out_size': self.out_size})
        return config

