---
layout: post
title: "Python Command Line One-Liners"
description: "Some quick ways to print Python from the command line"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/barn_owl_flying.jpg"
tags: [Linux, Python, Windows]
---

Sometimes I find it useful to be able to run Python commands right from the command line (without entering a Python console). Here are some ways I've found it useful.

> Note: If you have Windows, you'll need to use double quotes for these commands. For Unix-based systems (Mac and Linux), either single or double quotes should work.

## Print Hello World

`python -c "print('Hello World')"`

## Check if GPUs are available

`python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"`

## Find Package Version and Location

I also use this to find versions and file locations:
`python -c "import numpy; print(numpy.__version__); print(numpy.__file__)"`

## Print Model Details

You can even print out an entire machine learning model (It may download the first time if you don't already have it saved).

TensorFlow 2.X version:

`python -c "from tensorflow.keras.applications.vgg16 import VGG16; VGG16().summary()"`

Keras version:

`python -c "from keras.applications.vgg16 import VGG16; VGG16().summary()"`