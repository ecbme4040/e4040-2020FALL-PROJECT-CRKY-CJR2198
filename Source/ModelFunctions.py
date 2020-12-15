from tensorflow.keras import layers
import tensorflow as tf


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



def convolution(x_in, kernal_dim, num_kernals, batch_norm, strides, activation, kernel_initializer, gamma_init):
    x_out = layers.Conv2D(filters=num_kernals, kernel_size=kernal_dim, strides=strides,
                                   kernel_initializer=kernel_initializer,
                                   use_bias=False,
                                   padding='same')(x_in)

    if batch_norm == True:
        x_out = layers.BatchNormalization(gamma_initializer=gamma_init) (x_out)

    x_out = apply_activation(x_out, activation)

    return x_out

def deconvolution(x_in, kernal_dim, num_kernals, batch_norm, strides, activation, kernel_initializer, gamma_init):

    x_out = layers.Conv2DTranspose(filters=num_kernals, kernel_size=kernal_dim, strides=strides,
                                   kernel_initializer=kernel_initializer,
                                   use_bias=False,
                                   padding='same')(x_in)

    if batch_norm == True:
        x_out = layers.BatchNormalization(gamma_initializer=gamma_init) (x_out)

    x_out = apply_activation(x_out, activation)
    return x_out

def apply_activation(x_in, activation):

    if activation == 'selu':
        x_out = layers.Activation("selu")(x_in)
    if activation == "relu":
        x_out = layers.ReLU()(x_in)
    if activation == "leakyrelu":
        x_out = layers.LeakyReLU(0.2)(x_in)
    if activation == "tanh":
        x_out = layers.Activation("tanh")(x_in)

    return x_out


def create_generator_dict(gen_dict):
    #creates the generator from the supplied dictionary
    weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)

    num_latent = gen_dict['num_latent']
    dense_units = gen_dict['dense']
    activation = gen_dict['activation']
    filters = gen_dict['filter_size']
    filter_num = gen_dict['filter_num']
    strides = gen_dict['strides']

    latent_input = layers.Input(shape=(num_latent,))
    x = layers.Dense(dense_units[0]*dense_units[1]*dense_units[2], use_bias=True)(latent_input)
    x = layers.BatchNormalization(gamma_initializer=gamma_init)(x)
    x = apply_activation(x, activation)
    x = layers.Reshape((dense_units[0], dense_units[1], dense_units[2]))(x)

    for idx in range(len(filters)):
        x = deconvolution(x_in=x, kernal_dim=filters[idx], num_kernals=filter_num[idx], batch_norm=True, strides=strides[idx], activation=activation, kernel_initializer=weight_init, gamma_init=gamma_init)
    x = deconvolution(x_in=x, kernal_dim=5, num_kernals=gen_dict['out_channels'], batch_norm=False, strides=1, activation='tanh', kernel_initializer=weight_init, gamma_init=gamma_init) #no batch norm in the final layer

    model = tf.keras.Model(latent_input, x)

    return model

def create_discriminator_dict(disc_dict, input_dim):
    weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=1.0, stddev=0.02)

    activation = disc_dict['activation']
    filters = disc_dict['filter_size']
    filter_num = disc_dict['filter_num']
    dropout = disc_dict['dropout']
    noise = disc_dict['random_noise']

    x_in = layers.Input(shape=[input_dim[1], input_dim[2], input_dim[3]])

    for idx in range(len(filters)):
        if idx == 0:
            x = convolution(x_in, filters[idx], filter_num[idx], False, 2, activation, weight_init, gamma_init)
        else:
            x = convolution(x, filters[idx], filter_num[idx], True, 2, activation, weight_init, gamma_init)
        #if dropout > 0:
        #    x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)

    if dropout > 0:
     x = layers.Dropout(dropout)(x)

    if noise == True:
        x = layers.GaussianNoise(0.02)

    x = layers.Dense(1)(x)
    model = tf.keras.Model(x_in, x)

    return model
