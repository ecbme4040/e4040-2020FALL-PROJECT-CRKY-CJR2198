# e4040-2020Fall-project
Repository for CRKY Team Project (CJR2198)

![alt text](https://github.com/ecbme4040/e4040-2020FALL-PROJECT-CRKY-CJR2198/blob/main/Generated%20Examples/CelebAFinal.png?raw=true)

The paper this project is based on is "Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks" by Alec Radford, Luke Metz & Soumith Chintala (https://arxiv.org/abs/1511.06434)

The notebooks listed below contain a sample of the experiments referenced in the project report.

# Generation / Paper Reproduction Notebooks

ImageNet-1K CIFAR-10 Classification - I train a DCGAN on the ImageNet-1K dataset then use the descriminator as a feature extractor on CIFAR-10 and classify the images with an L2-
SVM

Generate ImageNet Images - Demonstration of ImageNet1k image generation with a trained DCGAN

Generate Face Images - Demonstration of facial image generation and vector arithmetic with trained DCGAN

# Training and Generation Notebooks

CelebA 128x128 - Scaling DCGAN architectures to higher resolutions

CelebA 64x64 - Sample of runs at 64x64 resolution with varying architectures

CelebA 32x32 - Sample of runs at 32x32 resolution with varying architectures

LSUN 64x64 - Sample of runs at 64x64 resolution with varying architectures

# MNIST DCGANs as Labeled Image Generators
MNIST Experiment - DCGANs are trained using a subset of training data, generators are used to generate training data and an EfficientNetB0 is trained using the generated data. Full dataset with augmentation used as benchmark.



The source files for the DCGAN model and model functions can be found within the SourceFiles/ModelSourceFiles subdirectory. Utility functions used for graphs can be found within the SourceFiles/Utilites subdirectory.


# Organization of this directory
To be populated by students, as shown in previous assignments
