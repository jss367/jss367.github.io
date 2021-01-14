---
layout: post
title: "Saving and Loading Models in TensorFlow"
description: "This post demonstrates how to save and reload TensorFlow models"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/ki_kangs.jpg"
tags: [Python, Computer Vision, TensorFlow, Neural Networks, Machine Learning, Convolutional Neural Networks]
---

There are many ways to save and load models in TensorFlow and Keras. It's good to have a range of options but sometimes with all of the flexibility it gets confusing which one you actually need in the moment. This post demonstrates the different methods available and talks about the strengths of each.

> Note: This post has been updated to use TensorFlow 2.

<b>Table of contents</b>
* TOC
{:toc}


```python
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.optimizers import Adam
```

This part is added to prevent the GPU from running out of memory (good for those with small GPUs).


```python
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
```

We'll use the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) as an example because loading it is already built into TensorFlow.


```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0
input_shape = train_images[0].shape
print(f'The image shape is {input_shape}')
```

    The image shape is (32, 32, 3)
    

Let's build a simple model. First we'll set some basic hyperparameters.


```python
BATCH_SIZE = 64
NUM_CLASSES = 10
NUM_EPOCHS = 2
```


```python
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
```

Now let's take a look at our model.


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 30, 30, 32)        896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 4, 4, 128)         73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 2, 2, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               131328    
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 258,762
    Trainable params: 258,762
    Non-trainable params: 0
    _________________________________________________________________
    

Let's look at the statistical distribution of the weights of the first layer. These are the weights that have been randomly (Xavier) initialized.


```python
orig_weights, orig_biases = model.layers[0].get_weights()
```


```python
def get_stats(nparray):
    print("Hash Value: ", hash(nparray.tobytes()))
    print("Shape: ", nparray.shape)
    print("Mean: ", np.mean(nparray))
    print("Standard Deviation: ", np.std(nparray))
    print("Min: ", np.min(nparray))
    print("Max: ", np.max(nparray))
```


```python
get_stats(orig_weights)
```

    Hash Value:  -2219139956115034589
    Shape:  (3, 3, 3, 32)
    Mean:  0.00071600085
    Standard Deviation:  0.07970445
    Min:  -0.13751489
    Max:  0.13770346
    

Now we'll need to set up a directory to store our model results.


```python
results_path = Path(r'C:\Users\Julius\Documents\GitHub\cv\results\Cifar10_' + datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(results_path, exist_ok=True)
```

We'll save the checkpoint as an HDF5 file. Note that you'll see both the `.hdf5` extension and the `.h5` extension.


```python
# Save the model according to the conditions

checkpoint_path = results_path / "model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', verbose=1, save_best_only=False, mode='auto', save_freq='epoch')
```


```python
optimizer = Adam()
loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
```


```python
model.compile(optimizer=optimizer,
              loss=loss_function,
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS, callbacks=[checkpoint],
                    validation_data=(test_images, test_labels))
```

    Epoch 1/2
    1561/1563 [============================>.] - ETA: 0s - loss: 2.1309 - accuracy: 0.3173
    Epoch 00001: saving model to C:\Users\Julius\Documents\GitHub\cv\results\Cifar10_20210113-001534\model-01-0.44.hdf5
    1563/1563 [==============================] - 28s 18ms/step - loss: 2.1309 - accuracy: 0.3173 - val_loss: 2.0136 - val_accuracy: 0.4406
    Epoch 2/2
    1561/1563 [============================>.] - ETA: 0s - loss: 2.0037 - accuracy: 0.4527
    Epoch 00002: saving model to C:\Users\Julius\Documents\GitHub\cv\results\Cifar10_20210113-001534\model-02-0.45.hdf5
    1563/1563 [==============================] - 29s 19ms/step - loss: 2.0037 - accuracy: 0.4526 - val_loss: 2.0062 - val_accuracy: 0.4499
    

Now let's look at the model weights.


```python
trained_weights, trained_biases = model.layers[0].get_weights()
```


```python
get_stats(trained_weights)
```

    Hash Value:  -2612516346418099248
    Shape:  (3, 3, 3, 32)
    Mean:  -0.0068695764
    Standard Deviation:  0.10035385
    Min:  -0.24802998
    Max:  0.23082884
    


```python
loss, accuracy = model.evaluate(test_images, test_labels, verbose=0)
accuracy
```




    0.4499000012874603




```python
model_predictions = model.predict(test_images, verbose=0)
```

The accuracy matches the final validation accuracy we got when training the model, as expected.

## Save Model with Keras .save method

There are lots of ways to save TensorFlow and Keras models. For Keras models, the simplest way is to use Keras's `.save` method.

This includes the training configuration (loss, optimizer).
The keras save method includes the architecture, weights, and state of the optimizer, so it's easy to resume training.

It can be saved to a Tensorflow SavedModel of a HDF5 file. These can both save the entire model to disk.


```python
model_filename = 'cifar10_model.hdf5'
```


```python
model.save(model_filename)
```

## Load Model with Keras .save method


```python
loaded_model = tf.keras.models.load_model(model_filename)
```

It is loaded and ready to go! We could use it for inference or keep training it.


```python
loaded_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 30, 30, 32)        896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 4, 4, 128)         73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 2, 2, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               131328    
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 258,762
    Trainable params: 258,762
    Non-trainable params: 0
    _________________________________________________________________
    

Note that we can't use `model.evalute` because the `evaluate` method relies on the metrics passed in the `model.compile` phase, and we haven't done that for our loaded model. If we try it the results will be the same as random guessing.


```python
loss, accuracy = loaded_model.evaluate(test_images, test_labels, verbose=0)
accuracy
```




    0.07490000128746033



But we can make predictions are ensure that those are the same.


```python
loaded_model_predictions = loaded_model.predict(test_images, verbose=0)
```


```python
(model_predictions == loaded_model_predictions).all()
```




    True




```python
loaded_weights, loaded_biases = loaded_model.layers[0].get_weights()
```

We can check to make sure the weights are identical.


```python
get_stats(trained_weights)
```

    Hash Value:  -2612516346418099248
    Shape:  (3, 3, 3, 32)
    Mean:  -0.0068695764
    Standard Deviation:  0.10035385
    Min:  -0.24802998
    Max:  0.23082884
    

And we can also check that is has the same optimizer.


```python
loaded_model.optimizer
```




    <tensorflow.python.keras.optimizer_v2.adam.Adam at 0x22f8ff1a580>



We can see that it's the same. Let's train it some more.


```python
RETRAIN_EPOCHS = 2
```


```python
loaded_history = loaded_model.fit(train_images, train_labels, epochs=RETRAIN_EPOCHS, callbacks=[checkpoint],
                    validation_data=(test_images, test_labels))
```

    Epoch 1/2
    1563/1563 [==============================] - ETA: 0s - loss: 1.9546 - accuracy: 0.0989
    Epoch 00001: saving model to C:\Users\Julius\Documents\GitHub\cv\results\Cifar10_20210113-001534\model-01-0.12.hdf5
    1563/1563 [==============================] - 26s 17ms/step - loss: 1.9546 - accuracy: 0.0989 - val_loss: 1.9459 - val_accuracy: 0.1238
    Epoch 2/2
    1562/1563 [============================>.] - ETA: 0s - loss: 1.9141 - accuracy: 0.0960
    Epoch 00002: saving model to C:\Users\Julius\Documents\GitHub\cv\results\Cifar10_20210113-001534\model-02-0.05.hdf5
    1563/1563 [==============================] - 26s 17ms/step - loss: 1.9141 - accuracy: 0.0960 - val_loss: 1.9254 - val_accuracy: 0.0472
    

## Save architecture and weight separately

This is quite common if you don't need all the optimizer details, like if you just wanted to run inference. Or if you were doing transfer learning.


```python
# serialize model to JSON
model_json = loaded_model.to_json()
with open(results_path / "model.json", "w") as json_file:
    json_file.write(model_json)
```

You can see that it's an entire json string.


```python
model_json
```




    '{"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 32, 32, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "batch_input_shape": [null, 32, 32, 3], "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": [3, 3], "strides": [1, 1], "padding": "valid", "data_format": "channels_last", "dilation_rate": [1, 1], "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_2", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "keras_version": "2.4.0", "backend": "tensorflow"}'



You could also do this with a yaml


```python
# serialize model to JSON
model_yaml = loaded_model.to_yaml()
with open(results_path / "model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
```

## Reload Weights


```python
model_json = Path(results_path / 'model.json').read_text()
```


```python
loaded_model = model_from_json(model_json)
```


```python
loaded_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 30, 30, 32)        896       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 4, 4, 128)         73856     
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 2, 2, 128)         0         
    _________________________________________________________________
    flatten (Flatten)            (None, 512)               0         
    _________________________________________________________________
    dense (Dense)                (None, 256)               131328    
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               32896     
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 258,762
    Trainable params: 258,762
    Non-trainable params: 0
    _________________________________________________________________
    


```python
loaded_weights, loaded_biases = loaded_model.layers[0].get_weights()
```


```python
get_stats(loaded_weights)
```

    Hash Value:  7420225496994056709
    Shape:  (3, 3, 3, 32)
    Mean:  -0.005051326
    Standard Deviation:  0.07917887
    Min:  -0.13790607
    Max:  0.13799374
    

Now load the weights


```python
saved_weights = checkpoint_path.parent / 'model-01-0.12.hdf5'
```


```python
loaded_model.load_weights(str(saved_weights))
```


```python
loaded_trained_weights, loaded_trained_biases = loaded_model.layers[0].get_weights()
```


```python
get_stats(loaded_trained_weights)
```

    Hash Value:  7653514471034317339
    Shape:  (3, 3, 3, 32)
    Mean:  -0.007722103
    Standard Deviation:  0.10606574
    Min:  -0.26978025
    Max:  0.25064382
    


```python
get_stats(trained_weights)
```

    Hash Value:  -2612516346418099248
    Shape:  (3, 3, 3, 32)
    Mean:  -0.0068695764
    Standard Deviation:  0.10035385
    Min:  -0.24802998
    Max:  0.23082884
    
