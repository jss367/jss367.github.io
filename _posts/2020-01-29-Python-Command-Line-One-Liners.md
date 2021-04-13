---
layout: post
title: "Python Command Line One-Liners"
description: "Some quick ways to print Python from the command line"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/rainbow.jpg"
tags: [Linux, Python, Windows]
---

Sometimes I find it useful to be able to run Python commands right from the command line (without entering a Python console). Here are some ways I've found it useful.

## Print Python from the command line

`python -c 'print("hello")'`

But on Windows you'll need double quotes

`python -c "print('hello')"`

You can even print out an entire machine learning model (It may download the first time if you don't already have it saved).

TF2.X version:

`python -c "from tensorflow.keras.applications.vgg16 import VGG16; print(VGG16().summary())"`

Keras version:

`python -c "from keras.applications.vgg16 import VGG16; print(VGG16().summary())"`

#### Check the GPUs are available

python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"
