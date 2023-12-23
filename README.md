# Explainable AI (XAI) via Prototypes

## Overview

This is a quick PyTorch implementation of the model and experiment presented in [Li et al.](https://arxiv.org/abs/1710.04806) in PyTorch.

This paper introduces __prototype neural networks__, a white-box autoencoder neural network structure. The learned network naturally comes with explanations for each prediction, and the explanations are loyal to what the network actually computes.

## Prototype Neural Networks

![Prototype Structure](https://github.com/1yian/mnist_prototypes/blob/master/prototype_struc.png?raw=true)

The prototype neural network structure integrates a prototype layer into a conventional autoencoder. Each prototype in the network is a representative example of a data point. When a new input is presented, the network compares it against these prototypes to make a decision. This approach aids in Explainable Artificial Intelligence (XAI) by providing clear, interpretable reasons for its decisions. Instead of being a 'black box', the network's reasoning becomes transparent, as it bases its decisions on similarities to known, understandable prototypes.

## Qualitative Results

I trained a prototype neural network on the simple MNIST dataset using the same training configuration as in [Li et al.](https://arxiv.org/abs/1710.04806). Below I show a random sample of the test set (first row), the autoencoder's reconstruction of the sample (second row), and the closest prototype to that sample (third row).

![Results](https://github.com/1yian/mnist_prototypes/blob/master/result.png?raw=true)

This illustrates how the prototype-based neural network processes and classifies digits. It learns prototypes, which are idealized representations of each class, and for any given input digit, the network identifies the most similar prototype. The close resemblance between the reconstructions and the prototypes underlines the interpretability aspect of the network, showcasing how decisions are made based on learned representative examples.

## Installation
Ensure you have the following packages installed properly:
```
torch
scikit-learn
numpy
tqdm
matplotlib
jupyter
```

## Usage

Training a new model can be done by running
```
python main.py
```
All hyperparameters are hardcoded to the ones suggested by [Li et al.](https://arxiv.org/abs/1710.04806).

Visualizing results of your model can be done in `Visualizations.ipynb`
