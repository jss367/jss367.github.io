---
layout: post
title: "CUDA Notes"
feature-img: "assets/img/rainbow.jpg"
tags: [CUDA, TensorFlow, PyTorch]
---



## Test if Tensorflow is working on the GPU

You can see all your physical devices like so:
``` python
import tensorflow as tf
tf.config.experimental.list_physical_devices()
```
and you can limit them to the GPU:
``` python
tf.config.experimental.list_physical_devices('GPU')
```
``` python
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```
