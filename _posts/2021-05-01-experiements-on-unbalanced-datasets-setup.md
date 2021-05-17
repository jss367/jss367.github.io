---
layout: post
title: "Experiments on Unbalanced Datasets - Setup"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/sun_tas.jpg"
tags: [Deep Learning, Python, TensorFlow]
---

In this series of posts I'm going to look at different methods of dealing with unbalanced data. Unbalanced datasets are common (and unavoidable) in the real world but I think don't get enough attention in the literature, where balanced datasets (e.g. MNIST, CIFAR 10) are too often used. Given an unbalanced real-world problem, we'll discuss on to set up the training, validation, and testing sets.

This post is the first in a series on working with unbalanced data. We'll answer questions like how to train a model, how to validate it, and how to test it. Is it better than your datasets be balanced or representative of the real-world distribution?

For these posts, we'll use the Kaggle [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats). The dataset has the same number of cat images as dog images, so we'll have to subset the dataset to run the experiments.  We’re going to pretend that there are 10 times as many cats as there are dogs in the “real world” population. Our goal is to build a model that answers the question, "Is this an image of a dog?"

<b>Table of contents</b>
* TOC
{:toc}


```python
import os
from os import listdir
from pathlib import Path
from typing import List

import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
from tensorflow.keras import metrics
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.models import Sequential
```

Because we'll be working with images, I'm including this to make sure my GPU doesn't run out of memory.


```python
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.random.set_seed(42)
```

## Prepare the Data

OK, now let's look at the dataset.


```python
image_dir = Path('E:/Data/Raw/cats_vs_dogs_dataset/all')
```


```python
class_names = listdir(image_dir)
print("Classes:\n", class_names)
```

    Classes:
     ['cats', 'dogs']
    

Let's see how many images we have.


```python
cat_dir = image_dir / 'cats'
dog_dir = image_dir / 'dogs'
```


```python
num_cat_train_im = len(listdir(cat_dir))
num_dog_train_im = len(listdir(dog_dir))
print(f"There are a total of {num_cat_train_im} cat images in the entire dataset.")
print(f"There are a total of {num_dog_train_im} dog images in the entire dataset.")
```

    There are a total of 5000 cat images in the entire dataset.
    There are a total of 5000 dog images in the entire dataset.
    

Now let's turn them into tf.data datasets.


```python
cat_list_ds = tf.data.Dataset.list_files(str(cat_dir/'*'), shuffle=False, seed=42)
dog_list_ds = tf.data.Dataset.list_files(str(dog_dir/'*'), shuffle=False, seed=42)
```

We have the same number of each dataset. We’ll have to build a function that creates unbalanced subsets of the dataset.


```python
def subset_dataset(dataset: tf.data.Dataset, splits: List) -> List[tf.data.Dataset]:
    """
    Split a dataset into any number of non-overlapping subdatasets of size listed in `splits`
    """
    assert sum(splits) <= tf.data.experimental.cardinality(dataset).numpy(), "Total number of images in splits exceeds dataset size"
    datasets = []
    total_used = 0
    for i, val in enumerate(splits):
        ds = dataset.skip(total_used).take(val)
        total_used += val
        datasets.append(ds)

    return datasets
```


```python
BATCH_SIZE = 32
NUM_EPOCHS = 20
img_height = 64
img_width = 64
num_channels = 3
```


```python
def prep_image(filename):
    img = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(img, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image = tf.image.resize(image, (img_height, img_width))
    return image

def prep_label(filename):
    parts = tf.strings.split(filename, sep=os.path.sep)
    one_hot_label = parts[-2] == class_names
    label = tf.argmax(one_hot_label)
    return label

def parse_file(filename):
    image = prep_image(filename)
    label = prep_label(filename)
    return image, label

def prepare_dataset(*datasets):
    dataset = tf.data.experimental.sample_from_datasets(datasets, seed=42)
    dataset = dataset.map(parse_file)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset
```

## Determine Metrics

Now we’ll have to decide on what metrics to use. We’ll want a variety of metrics to really explore what’s going on. Because the goal of this model is to find the dog images in the sea of cat images, we’ll consider a **true positive** to be correctly identifying an image of a **dog**. Correctly identifying a **cat** image will be considered a **true negative**.


```python
all_metrics = [
      metrics.TruePositives(name='tp'),
      metrics.FalsePositives(name='fp'),
      metrics.TrueNegatives(name='tn'),
      metrics.FalseNegatives(name='fn'), 
      metrics.BinaryAccuracy(name='accuracy'),
      metrics.Precision(name='precision'),
      metrics.Recall(name='recall'),
]
```

## Create Model

OK, now we have to make a model. This post doesn’t focus on the model so a simple CNN is all we'll need.


```python
def get_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='he_uniform', input_shape=(img_height, img_width, num_channels), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(64, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))
    model.add(Conv2D(128, kernel_size=(3,3), padding='same', strides=(1,1), kernel_initializer='he_uniform', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    return model
```

## Compile Model


```python
model = get_model()
```


```python
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=all_metrics)
```

## Functions for Visualizing the Results

We're going to need some functions to visualize the results, so let's build those here.


```python
def plot_loss(history, label):
    """
    Plot the train and val loss from a TensorFlow train
    """
    plt.plot(history.epoch, history.history['loss'],
               label='Train ' + label)
    plt.plot(history.epoch, history.history['val_loss'],
               label='Val ' + label,
               linestyle="--")
    plt.xlabel('Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

def plot_cm(labels, predictions, p=0.5):
    """
    Plot a confusion matrix
    """
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.cm.rocket_r)
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Truth label')
    plt.xlabel('Predicted label')
    
def calc_f1(metrics):
    """
    Assumes metrics contains precision at index 6 and recall at index 7
    """
    precision = metrics[6]
    recall = metrics[7]
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
```

## Conclusion

Now everything is all set up. In the follow notebooks we'll run various experiments with this.
