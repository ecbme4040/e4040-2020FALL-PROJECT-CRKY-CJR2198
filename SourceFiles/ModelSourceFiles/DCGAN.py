
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
from SourceFiles.ModelSourceFiles.ModelFunctions import *
from SourceFiles.Utilities.Utils import *
import os


### DCGAN Model Class ###
# Implemented as derived class of tf.keras.Model so that I can make use of model.fit()
# Overwrites are written for initialization, compile and train step
# Class contains two tensorflow models, generator and discriminator

class DCGAN (tf.keras.Model):
    ## Init override, required input: generator_dict, discriminator_dict, batch_size, model name and distribution flag
    def __init__(self, gen_dict, disc_dict, batch_size, model_name, distribute, slow_train=True): #defualt to slow training
        super(DCGAN, self).__init__()
        self.gen_model = create_generator_dict(gen_dict)#create_generator(number_of_latent_var)
        self.disc_model = create_discriminator_dict(disc_dict, self.gen_model.output_shape)#create_discriminator(self.gen_model.output_shape)
        self.batch_size = batch_size #single batch size per device mandatory for distributed training
        self.global_batch_size = batch_size * 2 #global batch size for a total update = #number of devices * per device batch size I assume 2 gpus
        self.num_latent = gen_dict['num_latent']# Number of latent variables to use in generator input
        self.model_name = model_name# Name of the model (used for saving routines)
        self.distribute = distribute# Distribute flag for running in distributed mode
        self.random_dist = gen_dict['latent_distribution'] # The distribution to sample from for the latent space (normal or uniform)

        ## Metric variables for reporting training progress ##
        self.Disc_gen_accuracy_metric = tf.keras.metrics.BinaryAccuracy() ## disc accuracy vs gen images
        self.Disc_real_accuracy_metric = tf.keras.metrics.BinaryAccuracy() ## disc accuracy vs real images
        self.Disc_binary_cross = tf.keras.metrics.BinaryCrossentropy() ## Loss metric for discriminator
        self.Gen_binary_cross = tf.keras.metrics.BinaryCrossentropy() ## Loss metric for generator


        self.slow_train = slow_train
        print(self.gen_model.output_shape)

    # Override compile method to take a separate optimizer for the generator and for the discriminator
    def compile(self, gen_opt, disc_optim, loss_obj):
        super(DCGAN, self).compile()
        self.disc_optimizer = disc_optim #Optimizer for discriminator model
        self.gen_optimizer = gen_opt #Optimizer for generator model
        self.loss_function = loss_obj #Loss object / function to use for model training

        # define model paths to save model files and examples
        os.mkdir(self.model_name)
        self.root_path = self.model_name + str('/')
        self.weight_path_gen = self.root_path + 'gen_weights'
        self.weight_path_disc = self.root_path + 'disc_weights'
        self.example_path = self.root_path + 'examples'
        self.log_path = self.root_path + 'logs'

        # create all necessary folders
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
        ## We want to use the generator to generate a sample not for update, set training to false
        images = self.gen_model(random_gen, training=False)
        images = images.numpy()
        fig = plt.figure(figsize=(12, 12))
        ## Code is similar to that in Tensorflow example ('https://www.tensorflow.org/tutorials/generative/dcgan') except I am using display to clear the notebook display
        ## I am regenerating new random samples every epoch, not reusing samples so that I can observe coverage quality
        ## For my implementation I had to cast generated images to numpy, convert to np.uint8 so that I can display RGB images

        display.clear_output(wait=True) ## Clear the notebook images
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((images[i, :, :, :] * 127.5 + 127.5).astype(np.uint8)) ## cast to normal (0,255) by multipying by 127.5 and adding 127.5 because generator output is same range as tanh
            plt.axis('off') ## remove the axis from the image
        fig.tight_layout() ##Tight layout for a nicer image
        plt.show()## show in the notebook
        plt.savefig(self.example_path + str('epoch_{:04d}.png'.format(epoch))) ## save it to the examples folder in model subfolder

    def save_model_weights(self, epoch):
        ## Saves model weights every epoch
        ## Dedicated folder for generator weights and discriminator weights
        self.gen_model.save(self.weight_path_gen + str('gen_model_epoch_{:04d}.h5'.format(epoch)))
        self.disc_model.save(self.weight_path_disc + str('disc_model_epoch_{:04d}.h5'.format(epoch)))

    def load_model_weights(self, gen_path, disc_path):
        ## Loads model weights specified in path
        self.gen_model.load_weights(gen_path)
        self.disc_model.load_weights(disc_path)

    # Method to generate a batch of random latent variables to be used by generator to generate images
    def get_latent_vector(self, batch_size_z):
        #Random normal or uniform distribution
        #Distribution set by user at class creation
        if self.random_dist == 'normal':
            return tf.random.normal([batch_size_z, self.num_latent], stddev=0.2)
        else:
            return tf.random.uniform([batch_size_z, self.num_latent], minval=-1., maxval=1.)


    def train_step(self,batch): #keras calls this function for each batch, it has to return the loss/accuracy metrics

        ## Two training routines are available, a fast (but unstable) and a slower, but much more stable - default to stable which was used throughout my project
        if self.slow_train == True:
            return self.train_step_slow(batch)
        else:
            return self.train_step_fast(batch)

    # Training step override function
    def train_step_slow(self, batch):

        # Generate a batch of random latent variables to be used to update the discriminator weights
        random_gen = self.get_latent_vector(tf.shape(batch)[0])



        ## Step 2.1.1 in pseudocode ##
        ## Calculate loss and gradients of the discriminator on real images
        with tf.GradientTape() as discriminator_tape:
            real_predictions = self.disc_model(batch)
            # if using distributed training routine, loss aggregation needs to be manually defined
            # in the default case, average loss is used
            # for distributed processing I use the average per example loss
            if self.distribute == True:  # flag for distributed training, need to calc average loss manually
                disc_real_loss = tf.reduce_sum(self.loss_function(tf.ones_like(real_predictions), real_predictions)) * (1. / self.global_batch_size) #calc per example avg loss across global batch
            else:
                disc_real_loss = self.loss_function(tf.ones_like(real_predictions),
                                                    real_predictions)  # In single gpu, loss function will average by default

        self.Disc_real_accuracy_metric.update_state(tf.ones_like(real_predictions), tf.nn.sigmoid(
            real_predictions))  ##Update accuracy metric, must apply sigmoid because using logits




        ## Step 2.1.2 in pseudocode ##
        # Capture the gradients of the loss relative to the weights in the discriminator and apply weight updates
        discriminator_grads = discriminator_tape.gradient(disc_real_loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(discriminator_grads, self.disc_model.trainable_variables))

        ## Step 2.1.3 in pseudocode ##
        ## Calculate loss and gradients of the discriminator when classifying generated images
        with tf.GradientTape() as discriminator_tape2:
            gen_images = self.gen_model(random_gen)  ## Generate images using the generator
            gen_predictions = self.disc_model(gen_images)  ## Use discriminator to predict the images

            if self.distribute == True:
                disc_gen_loss = tf.reduce_sum(self.loss_function(tf.zeros_like(gen_predictions), gen_predictions)) * (1. / self.global_batch_size) #calc per example avg loss across global batch
            else:
                disc_gen_loss = self.loss_function(tf.zeros_like(gen_predictions), gen_predictions)

        # Capture the gradients of the loss relative to the weights in the discriminator and apply weight updates
        discriminator_grads2 = discriminator_tape2.gradient(disc_gen_loss, self.disc_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(discriminator_grads2, self.disc_model.trainable_variables))
        self.Disc_gen_accuracy_metric.update_state(tf.zeros_like(gen_predictions), tf.nn.sigmoid(gen_predictions))




        ## Step 2.1.5 in pseudocode ##
        ## Calculate loss and gradients of the generator using the discriminator to classify generated images
        with tf.GradientTape() as generator_tape:
            # Generate a new set of random variables
            random_gen = self.get_latent_vector(self.batch_size)
            # Get predictions of the discriminator using the generated samples
            gen_predictions = self.disc_model(self.gen_model(random_gen))
            if self.distribute == True: 
                gen_mod_loss = tf.reduce_sum(self.loss_function(tf.ones_like(gen_predictions), gen_predictions)) * (1. / self.global_batch_size)  #calc per example avg loss across global batch
            else:
                gen_mod_loss = self.loss_function(tf.ones_like(gen_predictions), gen_predictions)

        # Capture the gradients of the loss relative to the weights in the generator and apply weight updates
        generator_grads = generator_tape.gradient(gen_mod_loss, self.gen_model.trainable_variables)
        self.gen_optimizer.apply_gradients(zip(generator_grads, self.gen_model.trainable_variables))
        average_disc_loss = (disc_gen_loss + disc_real_loss) / 2  ## Average loss of discriminator for comparison purposes




        # Return losses and accuracy, Keras will save these metrics in the history and in tensorboard callbacks so I can graph them later
        return {"Generator_Loss": gen_mod_loss, "Discriminator_Real_Loss": disc_real_loss,
                "Discriminator_Generator_Loss": disc_gen_loss,
                "Average_Discriminator_Loss": average_disc_loss,
                "Discriminator_Generator_Accuracy": self.Disc_gen_accuracy_metric.result(),
                "Discriminator_Real_Accuracy": self.Disc_real_accuracy_metric.result()}

    # Fast train step has poor convergence
    def train_step_fast(self, batch):

        ## This training routine is written only for distributed GPUs and cannot be used for single GPU
        ## The implementation is very similar to that employed in tensorflow's guide: https://www.tensorflow.org/tutorials/generative/dcgan

        random_gen = self.get_latent_vector(tf.shape(batch)[0])
        ## Calculate loss and gradients for both discriminator and generator on real and generated images in a single step
        ## This is fast but leads to instability, I tested this routine as a way to optimize training but it didn't work
        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            real_predictions = self.disc_model(batch)#predict real images
            gen_images = self.gen_model(random_gen)#generate fake images
            gen_predictions = self.disc_model(gen_images) #predict generated images using disc

            disc_real_loss = tf.nn.compute_average_loss(
                self.loss_function(tf.ones_like(real_predictions), real_predictions),
                global_batch_size=self.global_batch_size) ## compute average disc loss on real images
            self.Disc_binary_cross.update_state(tf.ones_like(real_predictions), tf.math.sigmoid(real_predictions))

            disc_gen_loss = tf.nn.compute_average_loss(
                self.loss_function(tf.zeros_like(gen_predictions), gen_predictions),
                global_batch_size=self.global_batch_size) ## compute average disc loss on gen images
            self.Disc_binary_cross.update_state(tf.zeros_like(gen_predictions), tf.math.sigmoid(gen_predictions))

            gen_mod_loss = tf.nn.compute_average_loss(
                self.loss_function(tf.ones_like(gen_predictions), gen_predictions),
                global_batch_size=self.global_batch_size) ## compute average gen loss
            self.Gen_binary_cross.update_state(tf.ones_like(gen_predictions), tf.math.sigmoid(gen_predictions))
            disc_avg_loss = (disc_gen_loss + disc_real_loss)/2 ##Compute average Disc loss

        # Single update to discriminator and generator
        discriminator_grads = discriminator_tape.gradient(disc_avg_loss, self.disc_model.trainable_variables)
        generator_grads = generator_tape.gradient(gen_mod_loss, self.gen_model.trainable_variables)
        self.disc_optimizer.apply_gradients(zip(discriminator_grads, self.disc_model.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(generator_grads, self.gen_model.trainable_variables))


        return {"Generator_Loss": gen_mod_loss, "Discriminator_Real_Loss": disc_real_loss,
                "Discriminator_Generator_Loss": disc_gen_loss, "Average_Discriminator_Loss": disc_avg_loss}


    def create_feature_extractor(self):

        ##Creates and returns feature extractor from trained discriminator model
        if self.disc_model:
            model_layers = self.disc_model.layers
            output_layers = []

            # iterates through the layers of the model
            # if it finds the output of a conv block (leakyrelu) then it will apply max pool and save as output layer after flatten
            for layer_idx in range(len(model_layers)):
                if isinstance(model_layers[layer_idx], tf.keras.layers.LeakyReLU):
                    output_layers.append(
                        layers.Flatten()(layers.MaxPooling2D(pool_size=(2, 2))(self.disc_model.layers[layer_idx].output))) # max pool and flatten

            ## concatenate all the flattened outputs
            final_out = layers.Concatenate()(output_layers)
            FeatureExtractor = tf.keras.models.Model(inputs=self.disc_model.input, outputs=final_out) #create the model
            FeatureExtractor.summary()

            return FeatureExtractor # Return the feature extractor
        return None

#Custom callback, save models, show the current progress image and reset the metrics for next epoch
class SaveModelAndCreateImages (tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None): ##called at the end of the epoch
        self.model.save_model_weights(epoch) ##Save models (full model rather than weights)
        self.model.gen_and_show_img(epoch) ##Generate and show images during training
        self.model.Disc_gen_accuracy_metric.reset_states()
        self.model.Disc_real_accuracy_metric.reset_states()
        self.model.Disc_binary_cross.reset_states()
        self.model.Gen_binary_cross.reset_states()

