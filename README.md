# e4040-2020Fall-project
Repository for CRKY Team Project (CJR2198)

# Generated Sample from CelebA DCGAN:

![alt text](https://github.com/ecbme4040/e4040-2020FALL-PROJECT-CRKY-CJR2198/blob/main/Generated%20Examples/CelebAFinal.png?raw=true)

The paper this project is based on is "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz & Soumith Chintala (https://arxiv.org/abs/1511.06434)

The notebooks listed below contain a sample of the experiments referenced in the project report.

# Training and Generation Notebooks

* CelebA 128x128 - Scaling DCGAN architectures to higher resolutions

* CelebA 64x64 - Sample of runs at 64x64 resolution with varying architectures

* CelebA 32x32 - Sample of runs at 32x32 resolution with varying architectures

* LSUN 64x64 - Sample of runs at 64x64 resolution with varying architectures

# Image Generation Notebooks

* Generate ImageNet Images - Demonstration of ImageNet1k image generation with a trained DCGAN

* Generate Face Images - Demonstration of facial image generation and vector arithmetic with trained DCGAN

# L2-SVM Paper Reproduction Notebook

* ImageNet-1K CIFAR-10 Classification - I train a DCGAN on the ImageNet-1K dataset then use the descriminator as a feature extractor on CIFAR-10 and classify the images with an L2-
SVM

# MNIST DCGANs as Labeled Image Generator Notebook
* MNIST Experiment - DCGANs are trained using a subset of training data, generators are used to generate training data and an EfficientNetB0 is trained using the generated data. Full dataset with augmentation used as benchmark.

# Requirements
 * Tensorflow 2.2 for all notebooks except the MNIST notebook, which requires Tensorflow 2.3 (for dataset_from_directory)
 
 * Seaborn (for plots)
 
 * tensorflow_datasets (to download datasets)
 
 * CIFAR-10 SVM Notebook - Requires scikit-learn for the L2-SVM
 

# Source Files 

The source files for the DCGAN model and model functions can be found within the SourceFiles/ModelSourceFiles subdirectory. 

Utility functions used for graphs can be found within the SourceFiles/Utilites subdirectory.


# Generated Example Images

Images generated using trained DCGANs can be found in the Generated Examples subdirectory.

# Instructions 

### To generate images:

To generate images using a DCGAN, load the generator from the file.

Check the input shape to the model. This will be the number of latent variables. 

Inputs will be either tf.random.normal([#images_to_generate, #latent_variables], stddev=0.2) or tf.random.uniform([#images_to_generate, #latent_variables], minval=-1., maxval=1.)

Refer to the notebooks "Generate Face Images.ipynb" and "Generate ImageNet Images.ipynb" for worked examples.

### To train a DCGAN:

To train a DCGAN, see notebooks for detailed examples and documentation refer to the notebook "CelebA 64x64.ipynb' (all notebooks have clear instructions and examples).

Train input images are image vectors normalized to (-1,1), model is fit with a tensorflow dataset.

To initalize/train a model:

Specificy an optimizer for both the discriminator and the generator.

Specify the generator architecture using the generator dictionary (see examples in notebooks).

Specify the discriminator dictionary using the discriminator dictionary (see examples in notebooks).

Specify a tf.keras.losses.BinaryCrossentropy(from_logits=True) object. Set reduction=tf.keras.losses.Reduction.NONE for distributed training.

Create the DCGAN model by calling the constructor: 

model = DCGAN(Generator_dict, Discriminator_dict, batch_size, model_name (string), distributed_training_flag) set the distributed training flag to false to train on single gpu.

Compile the created DCGAN model by calling the compile function with: 

model.compile(generator_optimizer, discriminator_optimizer, loss_object)

Train the model by calling model.fit()


# Trained Models

Will be uploaded to lionmail share folder:  https://drive.google.com/drive/folders/1amC47ZMfYxdGNkIH49kE9E0fgeykaGgV?usp=sharing

Both discriminator and generator weights are stored separately (either as .h5 or as model folders) they can be opened in the normal fashion tf.keras.models.load_model()

# Organization of this directory

```
.
├── CelebA 128x128.ipynb
├── CelebA 32x32.ipynb
├── CelebA 64x64.ipynb
├── E4040.2020Fall.DCGAN.report.cjr2198.pdf
├── Generated Examples
│   ├── 32x32_40%Dropout_Epoch_60CelebA.png
│   ├── 32x32 CelebA 40% Dropout.png
│   ├── 32x32 CelebA No Dropout.png
│   ├── 32x32ImageNetEpoch10.png
│   ├── 32x32ImageNet Epoch255 Deep Generator.png
│   ├── 32x32ImageNetEpoch40.png
│   ├── 32x32ImageNetEpoch80.png
│   ├── 32x32ImageNet.png
│   ├── 32x32NoDropoutCeleb60Epoch.png
│   ├── 64x64 CelebA Deep Discriminator Epoch 37.png
│   ├── 64x64 CelebA Deep Discriminator Epoch 38.png
│   ├── 64x64CelebA_Normal_Distribution.png
│   ├── 64x64CelebA_Uniform_Distribution.png
│   ├── 64x64CelebA_with_bias.png
│   ├── 64x64 CelebA_without_bias.png
│   ├── 64x64 Celeb Experiment.png
│   ├── 64x64 Celeb High Quality.png
│   ├── 64x64 Celeb Normal.png
│   ├── 64x64 Celeb Uniform.png
│   ├── 64x64LSUN_256_Latent.png
│   ├── 64x64LSUN_normal_dist_256_latent.png
│   └── CelebAFinal.png
├── Generate Face Images.ipynb
├── Generate ImageNet Images.ipynb
├── .github
│   └── .keep
├── ImageNet-1K CIFAR-10 Classification.ipynb
├── .ipynb_checkpoints
├── LSUN 64x64.ipynb
├── MNIST Experiment.ipynb
├── README.md
└── SourceFiles
    ├── ModelSourceFiles
    │   ├── DCGAN.py
    │   ├── ModelFunctions.py
    │   └── Utils.py
    └── Utilities
        └── Utils.py

6 directories, 37 files
```
