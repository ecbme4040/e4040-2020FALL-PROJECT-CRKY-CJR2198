
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import tqdm
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time
from ModelFunctions import *
from Utils import *

import os


### DCGAN Model Class ###
# Implemented as derived class of tf.keras.Model so that I can make use of model.fit()
# Overwrites are written for initialization, compile and train step
# Class contains two tensorflow models, generator and discriminator

class DCGAN (tf.keras.Model):
    def __init__(self, gen_dict, disc_dict, batch_size, model_name, distribute, slow_train=True):
        super(DCGAN, self).__init__()
        self.gen_model = create_generator_dict(gen_dict)#create_generator(number_of_latent_var)
        self.disc_model = create_discriminator_dict(disc_dict, self.gen_model.output_shape)#create_discriminator(self.gen_model.output_shape)
        self.batch_size = batch_size #single batch size per device
        self.global_batch_size = batch_size * 2 #global batch size for a total update = #number of devices * per device batch size
        self.num_latent = gen_dict['num_latent']# Number of latent variables to use in generator input
        self.model_name = model_name# Name of the model (used for saving routines)
        self.distribute = distribute# Distribute flag for running in distributed mode
        self.random_dist = gen_dict['latent_distribution'] # The distribution to sample from for the latent space (normal or uniform)
        self.Disc_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        self.Disc_gen_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        self.Disc_real_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        self.Disc_binary_cross = tf.keras.metrics.BinaryCrossentropy()
        self.Gen_binary_cross = tf.keras.metrics.BinaryCrossentropy()
        self.Disc_loss = []
        self.Gen_loss = []

        self.slow_train = slow_train
        print(self.gen_model.output_shape)

    # Overwrite compile method to take a separate optimizer for the generator and for the discriminator
    def compile(self, gen_opt, disc_optim, loss_obj):
        super(DCGAN, self).compile()
        self.disc_optimizer = disc_optim #Optimizer for discriminator model
        self.gen_optimizer = gen_opt #Optimizer for generator model
        self.loss_function = loss_obj #Loss object / function to use for model training

        # make model folder and subfolders, define model paths to save model files and examples
        os.mkdir(self.model_name)
        self.root_path = self.model_name + str('/')
        self.weight_path_gen = self.root_path + 'gen_weights'
        self.weight_path_disc = self.root_path + 'disc_weights'
        self.example_path = self.root_path + 'examples'
        self.log_path = self.root_path + 'logs'

        os.mkdir(self.weight_path_gen)
        os.mkdir(self.weight_path_disc)
        os.mkdir(self.example_path)
        os.mkdir(self.log_path)
        self.weight_path_gen += '/'
        self.weight_path_disc += '/'
        self.example_path += '/'
        self.log_path += '/'

    # Method to generate and show images during training
    def gen_and_show_img (self, epoch):
        random_gen = self.get_latent_vector(16)
        predictions = self.gen_model(random_gen, training=False)
        predictions = predictions.numpy()
        fig = plt.figure(figsize=(12, 12))
        display.clear_output(wait=True)
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i, :, :, :] * 127.5 + 127.5).astype(np.uint8))
            plt.axis('off')

        plt.savefig(self.example_path + str('epoch_{:04d}.png'.format(epoch)))
        #if epoch % 5 == 0:
        #    plt.savefig(self.example_path + str('epoch_{:04d}.png'.format(epoch)))

        plt.show()

    def save_model_weights(self, epoch):
        self.gen_model.save(self.weight_path_gen + str('gen_model_epoch_{:04d}.h5'.format(epoch)))
        self.disc_model.save(self.weight_path_disc + str('disc_model_epoch_{:04d}.h5'.format(epoch)))

    def load_model_weights(self, gen_path, disc_path):
        self.gen_model.load_weights(gen_path)
        self.disc_model.load_weights(disc_path)

    # Method to generate a batch of random latent variables to be used by generator to generate images
    def get_latent_vector(self, batch_size_z):
        #Random normal or uniform distribution
        if self.random_dist == 'normal':
            return tf.random.normal([batch_size_z, self.num_latent], stddev=0.2)
        else:
            return tf.random.uniform([batch_size_z, self.num_latent], minval=-1., maxval=1.)

    def train_step_fast(self, batch):

        random_gen = self.get_latent_vector(tf.shape(batch)[0])

        ## Calculate loss and gradients of the discriminator on real images
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            real_predictions = self.disc_model(batch)
            gen_images = self.gen_model(random_gen)
            gen_predictions = self.disc_model(gen_images)

            disc_real_loss = tf.nn.compute_average_loss(
                self.loss_function(tf.ones_like(real_predictions), real_predictions),
                global_batch_size=self.global_batch_size)
            self.Disc_binary_cross.update_state(tf.ones_like(real_predictions), tf.math.sigmoid(real_predictions))

            disc_gen_loss = tf.nn.compute_average_loss(
                self.loss_function(tf.zeros_like(gen_predictions), gen_predictions),
                global_batch_size=self.global_batch_size)
            self.Disc_binary_cross.update_state(tf.zeros_like(gen_predictions), tf.math.sigmoid(gen_predictions))

            gen_mod_loss = tf.nn.compute_average_loss(
                self.loss_function(tf.ones_like(gen_predictions), gen_predictions),
                global_batch_size=self.global_batch_size)
            self.Gen_binary_cross.update_state(tf.ones_like(gen_predictions), tf.math.sigmoid(gen_predictions))
            disc_avg_loss = (disc_gen_loss + disc_real_loss)/2

        discriminator_grads = discriminator_tape.gradient(disc_avg_loss, self.disc_model.trainable_variables)
        generator_grads = generator_tape.gradient(gen_mod_loss, self.gen_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(discriminator_grads, self.disc_model.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(generator_grads, self.gen_model.trainable_variables))

        #self.Disc_loss.append(self.Disc_binary_cross.result())
        #self.Gen_loss.append(self.Gen_binary_cross.result())


        return {"Generator_Loss": gen_mod_loss, "Discriminator_Real_Loss": disc_real_loss,
                "Discriminator_Generator_Loss": disc_gen_loss, "Average_Discriminator_Loss": disc_avg_loss}


    # Training step override function
    def train_step_slow(self, batch):

        # Generate a batch of random latent variables to be used to update the discriminator weights
        random_gen = self.get_latent_vector(tf.shape(batch)[0])

        ## Calculate loss and gradients of the discriminator on real images
        with tf.GradientTape() as discriminator_tape:
            real_predictions = self.disc_model(batch)
            # if using distributed training routine, loss aggregation needs to be manually defined
            # in the default case, average loss is used
            # for distributed processing I use the average per example loss
            if self.distribute == True: #flag for distributed training
                disc_real_loss = tf.reduce_sum(self.loss_function(tf.ones_like(real_predictions), real_predictions)) * (1. / self.global_batch_size)
                #disc_real_loss = tf.nn.compute_average_loss(self.loss_function(tf.ones_like(real_predictions), real_predictions),global_batch_size=self.global_batch_size)
            else:
                disc_real_loss = self.loss_function(tf.ones_like(real_predictions), real_predictions)

        self.Disc_real_accuracy_metric.update_state(tf.ones_like(real_predictions), tf.nn.sigmoid(real_predictions))

        # Capture the gradients of the loss relative to the weights in the discriminator and apply weight updates
        discriminator_grads = discriminator_tape.gradient(disc_real_loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(discriminator_grads, self.disc_model.trainable_variables))

        ## Calculate loss and gradients of the discriminator when classifying generated images
        with tf.GradientTape() as discriminator_tape2:
            gen_images = self.gen_model(random_gen)
            gen_predictions = self.disc_model(gen_images)

            if self.distribute == True:
                disc_gen_loss = tf.reduce_sum(self.loss_function(tf.zeros_like(gen_predictions), gen_predictions)) * (1. / self.global_batch_size)
                #disc_gen_loss = tf.nn.compute_average_loss(self.loss_function(tf.zeros_like(gen_predictions), gen_predictions), global_batch_size=self.global_batch_size)
            else:
                disc_gen_loss = self.loss_function(tf.zeros_like(gen_predictions), gen_predictions)

        # Capture the gradients of the loss relative to the weights in the discriminator and apply weight updates
        discriminator_grads2 = discriminator_tape2.gradient(disc_gen_loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(discriminator_grads2, self.disc_model.trainable_variables))

        self.Disc_gen_accuracy_metric.update_state(tf.zeros_like(gen_predictions), tf.nn.sigmoid(gen_predictions))


        ## Calculate loss and gradients of the generator using the discriminator to classify generated images
        with tf.GradientTape() as generator_tape:
            # Generate a new set of random variables
            random_gen = self.get_latent_vector(self.batch_size)
            # Get predictions of the discriminator using the generated samples
            gen_predictions = self.disc_model(self.gen_model(random_gen))
            if self.distribute == True:
                gen_mod_loss = tf.reduce_sum(self.loss_function(tf.ones_like(gen_predictions), gen_predictions)) * (1. / self.global_batch_size)  # self.calculate_gen_loss(gen_predictions)
                #gen_mod_loss = tf.nn.compute_average_loss(self.loss_function(tf.ones_like(gen_predictions), gen_predictions), global_batch_size=self.global_batch_size)
            else:
                gen_mod_loss = self.loss_function(tf.ones_like(gen_predictions), gen_predictions)

        # Capture the gradients of the loss relative to the weights in the generator and apply weight updates
        generator_grads = generator_tape.gradient(gen_mod_loss, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(generator_grads, self.gen_model.trainable_variables))

        average_disc_loss = (disc_gen_loss + disc_real_loss) / 2
        # Return 3 losses and accuracy
        return{"Generator_Loss": gen_mod_loss, "Discriminator_Real_Loss": disc_real_loss,
               "Discriminator_Generator_Loss": disc_gen_loss,
               "Average_Discriminator_Loss": average_disc_loss,
               "Discriminator_Generator_Accuracy":self.Disc_gen_accuracy_metric.result(),
               "Discriminator_Real_Accuracy":self.Disc_real_accuracy_metric.result()}

    def train_step(self,batch):

        if self.slow_train == True:
            return self.train_step_slow(batch)
        else:
            return self.train_step_fast(batch)

    def create_feature_extractor(self):

        ##Creates feature extractor from trained discriminator model
        if self.disc_model:
            model_layers = self.disc_model.layers
            output_layers = []
            for layer_idx in range(len(model_layers)):
                if isinstance(model_layers[layer_idx], tf.keras.layers.LeakyReLU):
                    output_layers.append(
                        layers.Flatten()(layers.MaxPooling2D(pool_size=(2, 2))(self.disc_model.layers[layer_idx].output)))

            final_out = layers.Concatenate()(output_layers)
            FeatureExtractor = tf.keras.models.Model(inputs=self.disc_model.input, outputs=final_out)
            FeatureExtractor.summary()

            return FeatureExtractor
        return None

    # def plot_loss(self):
    #     data = {"Discriminator Loss": , "K=15": k_15, "K=50": k_50, 'Iteration': index}
    #     log_data = pd.DataFrame(log_dat)
    #     fig, ax = plt.subplots(figsize=[8, 6])
    #     plot = sns.lineplot(data=pd.melt(log_data, ['Iteration']), y='value', x='Iteration', hue='variable')
    #     plot.set_title('Variational Inference Objective Function vs Iteration (4 to 1000)')
    #     plot.set_ylabel("VI Objective Function")
    #     plot.set_xlabel("Iteration Count")
    #     plot.legend_.set_title('')
    #     plt.show()

class SaveModelAndCreateImages (tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save_model_weights(epoch)
        self.model.gen_and_show_img(epoch)
        self.model.Disc_accuracy_metric.reset_states()
        self.model.Disc_binary_cross.reset_states()
        self.model.Gen_binary_cross.reset_states()

