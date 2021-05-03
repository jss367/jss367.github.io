---
layout: post
title: "Evaluation on Unbalanced Datasets"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/sun_tas.jpg"
tags: [Deep Learning, Python, TensorFlow]
---

In this post I'm going to look at different methods of evaluating models on unbalanced populations. This is a supplement to my post on training on onbalanced datasets. For this post, we'll use the Kaggle [Dogs vs. Cats dataset](https://www.kaggle.com/c/dogs-vs-cats). The dataset has the same number of cat images as dog images, so we'll have to subset the dataset to run the experiment. We're going to pretend that there are 10 times as many cats as there are dogs in our population, and we want to build a model that answers the question, "Is this an image of a dog?" Thus a true positive would be correctly identifying an image of a dog.

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

Because we'll be working with images, I'm going to make sure my GPU doesn't run out of memory.


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
root_dir = Path('E:/Data/Raw/cats_vs_dogs_dataset')
image_dir = root_dir / 'train'
```


```python
class_names = listdir(image_dir)
print(class_names)
```

    ['cats', 'dogs']
    

Let's see how many images we have.


```python
cat_dir = image_dir / 'cats'
dog_dir = image_dir / 'dogs'
```


```python
num_cat_train_im = len(listdir(cat_dir))
num_dog_train_im = len(listdir(dog_dir))
print(num_cat_train_im)
print(num_dog_train_im)
```

    4000
    4000
    

We have the same number of each dataset. Now, let's say we're looking for cats in a sea of dog images... how should we go about this?
Let's say we only have 50 cat images and 5000 dogs images.


```python
cat_list_ds = tf.data.Dataset.list_files(str(cat_dir/'*'), shuffle=False, seed=42)
dog_list_ds = tf.data.Dataset.list_files(str(dog_dir/'*'), shuffle=False, seed=42)
```


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

Now we'll have to decide on what metrics to use. These will be important so we'll use a lot of them.


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

OK, now we have to make a model. This post isn't about the model so I'm going to make a simple CNN.


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

## Visualize Results

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

## Train Model

OK. Out first experiment we'll make a couple train datasets. One options is to have a balanced dataset, the other is to allow it to be unbalanced to match the "real world". Let's see which one produces better results.


```python
cat_list_train, cat_list_val, cat_list_test_balanced, cat_list_test_unbalanced = subset_dataset(cat_list_ds, [1000, 1000, 1000, 1000])
dog_list_train, dog_list_val, dog_list_test_balanced, dog_list_test_unbalanced = subset_dataset(dog_list_ds, [1000, 100, 1000, 100])
```


```python
train_ds = prepare_dataset(cat_list_train, dog_list_train)
val_ds = prepare_dataset(cat_list_val, dog_list_val)
test_ds_balanced = prepare_dataset(cat_list_test_balanced, dog_list_test_balanced)
test_ds_unbalanced = prepare_dataset(cat_list_test_unbalanced, dog_list_test_unbalanced)
```

Great. Now let's train the models. We have two. One that likes cats more than dogs and one that likes dogs more than cats.


```python
cat_weights = {0:2, 1:1}
model_cat = get_model()
model_cat.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_cat = model_cat.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=cat_weights)
```

    Epoch 1/20
    63/63 [==============================] - 10s 113ms/step - loss: 1.6739 - tp: 94.3281 - fp: 89.8906 - tn: 409.1562 - fn: 445.6250 - accuracy: 0.4796 - precision: 0.5071 - recall: 0.2304 - val_loss: 0.4007 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 2/20
    63/63 [==============================] - 5s 87ms/step - loss: 1.0227 - tp: 0.0000e+00 - fp: 0.0000e+00 - tn: 499.0469 - fn: 539.9531 - accuracy: 0.4757 - precision: 0.0000e+00 - recall: 0.0000e+00 - val_loss: 0.3462 - val_tp: 0.0000e+00 - val_fp: 0.0000e+00 - val_tn: 1000.0000 - val_fn: 100.0000 - val_accuracy: 0.9091 - val_precision: 0.0000e+00 - val_recall: 0.0000e+00
    Epoch 3/20
    63/63 [==============================] - 5s 87ms/step - loss: 1.0457 - tp: 8.1562 - fp: 3.2031 - tn: 495.8438 - fn: 531.7969 - accuracy: 0.4785 - precision: 0.3544 - recall: 0.0089 - val_loss: 0.4189 - val_tp: 1.0000 - val_fp: 2.0000 - val_tn: 998.0000 - val_fn: 99.0000 - val_accuracy: 0.9082 - val_precision: 0.3333 - val_recall: 0.0100
    Epoch 4/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.9658 - tp: 10.0781 - fp: 4.2969 - tn: 494.7500 - fn: 529.8750 - accuracy: 0.4806 - precision: 0.6114 - recall: 0.0172 - val_loss: 0.3708 - val_tp: 2.0000 - val_fp: 2.0000 - val_tn: 998.0000 - val_fn: 98.0000 - val_accuracy: 0.9091 - val_precision: 0.5000 - val_recall: 0.0200
    Epoch 5/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.9542 - tp: 47.8906 - fp: 13.2344 - tn: 485.8125 - fn: 492.0625 - accuracy: 0.5005 - precision: 0.7921 - recall: 0.0629 - val_loss: 0.4169 - val_tp: 2.0000 - val_fp: 4.0000 - val_tn: 996.0000 - val_fn: 98.0000 - val_accuracy: 0.9073 - val_precision: 0.3333 - val_recall: 0.0200
    Epoch 6/20
    63/63 [==============================] - 5s 88ms/step - loss: 0.9102 - tp: 78.5469 - fp: 23.6094 - tn: 475.4375 - fn: 461.4062 - accuracy: 0.5132 - precision: 0.6803 - recall: 0.1065 - val_loss: 0.3159 - val_tp: 5.0000 - val_fp: 2.0000 - val_tn: 998.0000 - val_fn: 95.0000 - val_accuracy: 0.9118 - val_precision: 0.7143 - val_recall: 0.0500
    Epoch 7/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.9252 - tp: 44.5625 - fp: 9.4844 - tn: 489.5625 - fn: 495.3906 - accuracy: 0.5038 - precision: 0.8697 - recall: 0.0663 - val_loss: 0.3269 - val_tp: 13.0000 - val_fp: 10.0000 - val_tn: 990.0000 - val_fn: 87.0000 - val_accuracy: 0.9118 - val_precision: 0.5652 - val_recall: 0.1300
    Epoch 8/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.8638 - tp: 176.8594 - fp: 44.7812 - tn: 454.2656 - fn: 363.0938 - accuracy: 0.5907 - precision: 0.8062 - recall: 0.2927 - val_loss: 0.3711 - val_tp: 32.0000 - val_fp: 45.0000 - val_tn: 955.0000 - val_fn: 68.0000 - val_accuracy: 0.8973 - val_precision: 0.4156 - val_recall: 0.3200
    Epoch 9/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.8401 - tp: 229.1719 - fp: 58.0312 - tn: 441.0156 - fn: 310.7812 - accuracy: 0.6261 - precision: 0.8244 - recall: 0.3791 - val_loss: 0.2907 - val_tp: 24.0000 - val_fp: 18.0000 - val_tn: 982.0000 - val_fn: 76.0000 - val_accuracy: 0.9145 - val_precision: 0.5714 - val_recall: 0.2400
    Epoch 10/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.8075 - tp: 230.2188 - fp: 45.2812 - tn: 453.7656 - fn: 309.7344 - accuracy: 0.6415 - precision: 0.8649 - recall: 0.3841 - val_loss: 0.3356 - val_tp: 34.0000 - val_fp: 41.0000 - val_tn: 959.0000 - val_fn: 66.0000 - val_accuracy: 0.9027 - val_precision: 0.4533 - val_recall: 0.3400
    Epoch 11/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.7599 - tp: 272.3906 - fp: 66.3906 - tn: 432.6562 - fn: 267.5625 - accuracy: 0.6679 - precision: 0.8295 - recall: 0.4715 - val_loss: 0.3451 - val_tp: 30.0000 - val_fp: 51.0000 - val_tn: 949.0000 - val_fn: 70.0000 - val_accuracy: 0.8900 - val_precision: 0.3704 - val_recall: 0.3000
    Epoch 12/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.7339 - tp: 284.6562 - fp: 62.1406 - tn: 436.9062 - fn: 255.2969 - accuracy: 0.6684 - precision: 0.8331 - recall: 0.4691 - val_loss: 0.3195 - val_tp: 35.0000 - val_fp: 56.0000 - val_tn: 944.0000 - val_fn: 65.0000 - val_accuracy: 0.8900 - val_precision: 0.3846 - val_recall: 0.3500
    Epoch 13/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.6968 - tp: 296.8125 - fp: 49.5781 - tn: 449.4688 - fn: 243.1406 - accuracy: 0.6996 - precision: 0.8649 - recall: 0.5103 - val_loss: 0.3082 - val_tp: 37.0000 - val_fp: 48.0000 - val_tn: 952.0000 - val_fn: 63.0000 - val_accuracy: 0.8991 - val_precision: 0.4353 - val_recall: 0.3700
    Epoch 14/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.6173 - tp: 338.2344 - fp: 52.9844 - tn: 446.0625 - fn: 201.7188 - accuracy: 0.7401 - precision: 0.8814 - recall: 0.5898 - val_loss: 0.3456 - val_tp: 50.0000 - val_fp: 92.0000 - val_tn: 908.0000 - val_fn: 50.0000 - val_accuracy: 0.8709 - val_precision: 0.3521 - val_recall: 0.5000
    Epoch 15/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.5739 - tp: 370.7188 - fp: 46.7500 - tn: 452.2969 - fn: 169.2344 - accuracy: 0.7832 - precision: 0.9006 - recall: 0.6623 - val_loss: 0.2872 - val_tp: 41.0000 - val_fp: 53.0000 - val_tn: 947.0000 - val_fn: 59.0000 - val_accuracy: 0.8982 - val_precision: 0.4362 - val_recall: 0.4100
    Epoch 16/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.5674 - tp: 360.2344 - fp: 39.6719 - tn: 459.3750 - fn: 179.7188 - accuracy: 0.7645 - precision: 0.9144 - recall: 0.6143 - val_loss: 0.3166 - val_tp: 43.0000 - val_fp: 64.0000 - val_tn: 936.0000 - val_fn: 57.0000 - val_accuracy: 0.8900 - val_precision: 0.4019 - val_recall: 0.4300
    Epoch 17/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.5173 - tp: 385.9688 - fp: 36.0156 - tn: 463.0312 - fn: 153.9844 - accuracy: 0.8065 - precision: 0.9259 - recall: 0.6890 - val_loss: 0.3106 - val_tp: 38.0000 - val_fp: 72.0000 - val_tn: 928.0000 - val_fn: 62.0000 - val_accuracy: 0.8782 - val_precision: 0.3455 - val_recall: 0.3800
    Epoch 18/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.5211 - tp: 380.6406 - fp: 37.8594 - tn: 461.1875 - fn: 159.3125 - accuracy: 0.7932 - precision: 0.9230 - recall: 0.6654 - val_loss: 0.3425 - val_tp: 43.0000 - val_fp: 82.0000 - val_tn: 918.0000 - val_fn: 57.0000 - val_accuracy: 0.8736 - val_precision: 0.3440 - val_recall: 0.4300
    Epoch 19/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.4883 - tp: 398.2188 - fp: 39.2500 - tn: 459.7969 - fn: 141.7344 - accuracy: 0.8064 - precision: 0.9149 - recall: 0.6984 - val_loss: 0.4432 - val_tp: 60.0000 - val_fp: 151.0000 - val_tn: 849.0000 - val_fn: 40.0000 - val_accuracy: 0.8264 - val_precision: 0.2844 - val_recall: 0.6000
    Epoch 20/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.4241 - tp: 423.5781 - fp: 33.5938 - tn: 465.4531 - fn: 116.3750 - accuracy: 0.8471 - precision: 0.9326 - recall: 0.7650 - val_loss: 0.4485 - val_tp: 54.0000 - val_fp: 127.0000 - val_tn: 873.0000 - val_fn: 46.0000 - val_accuracy: 0.8427 - val_precision: 0.2983 - val_recall: 0.5400
    


```python
dog_weights = {0:1, 1:2}
model_dog = get_model()
model_dog.compile(optimizer='adam', loss='binary_crossentropy', metrics=all_metrics)
history_dog = model_dog.fit(train_ds, epochs=NUM_EPOCHS, validation_data=val_ds, class_weight=dog_weights)
```

    Epoch 1/20
    63/63 [==============================] - 8s 98ms/step - loss: 1.3437 - tp: 545.4062 - fp: 595.5781 - tn: 903.4688 - fn: 94.5469 - accuracy: 0.6923 - precision: 0.4605 - recall: 0.8026 - val_loss: 0.7640 - val_tp: 100.0000 - val_fp: 1000.0000 - val_tn: 0.0000e+00 - val_fn: 0.0000e+00 - val_accuracy: 0.0909 - val_precision: 0.0909 - val_recall: 1.0000
    Epoch 2/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.9900 - tp: 539.9531 - fp: 499.0469 - tn: 0.0000e+00 - fn: 0.0000e+00 - accuracy: 0.5243 - precision: 0.5243 - recall: 1.0000 - val_loss: 0.7313 - val_tp: 98.0000 - val_fp: 903.0000 - val_tn: 97.0000 - val_fn: 2.0000 - val_accuracy: 0.1773 - val_precision: 0.0979 - val_recall: 0.9800
    Epoch 3/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.9792 - tp: 520.1562 - fp: 464.9844 - tn: 34.0625 - fn: 19.7969 - accuracy: 0.5399 - precision: 0.5355 - recall: 0.9445 - val_loss: 0.7873 - val_tp: 95.0000 - val_fp: 806.0000 - val_tn: 194.0000 - val_fn: 5.0000 - val_accuracy: 0.2627 - val_precision: 0.1054 - val_recall: 0.9500
    Epoch 4/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.9483 - tp: 510.3594 - fp: 429.0781 - tn: 69.9688 - fn: 29.5938 - accuracy: 0.5676 - precision: 0.5522 - recall: 0.9367 - val_loss: 0.7147 - val_tp: 84.0000 - val_fp: 607.0000 - val_tn: 393.0000 - val_fn: 16.0000 - val_accuracy: 0.4336 - val_precision: 0.1216 - val_recall: 0.8400
    Epoch 5/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.9366 - tp: 506.2031 - fp: 417.8125 - tn: 81.2344 - fn: 33.7500 - accuracy: 0.5733 - precision: 0.5586 - recall: 0.9158 - val_loss: 0.6139 - val_tp: 53.0000 - val_fp: 266.0000 - val_tn: 734.0000 - val_fn: 47.0000 - val_accuracy: 0.7155 - val_precision: 0.1661 - val_recall: 0.5300
    Epoch 6/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.9376 - tp: 477.4688 - fp: 389.0781 - tn: 109.9688 - fn: 62.4844 - accuracy: 0.5646 - precision: 0.5604 - recall: 0.8273 - val_loss: 0.5685 - val_tp: 49.0000 - val_fp: 190.0000 - val_tn: 810.0000 - val_fn: 51.0000 - val_accuracy: 0.7809 - val_precision: 0.2050 - val_recall: 0.4900
    Epoch 7/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.9380 - tp: 466.1406 - fp: 353.3125 - tn: 145.7344 - fn: 73.8125 - accuracy: 0.5886 - precision: 0.5837 - recall: 0.7964 - val_loss: 0.6163 - val_tp: 68.0000 - val_fp: 351.0000 - val_tn: 649.0000 - val_fn: 32.0000 - val_accuracy: 0.6518 - val_precision: 0.1623 - val_recall: 0.6800
    Epoch 8/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.8616 - tp: 479.2969 - fp: 345.3438 - tn: 153.7031 - fn: 60.6562 - accuracy: 0.6061 - precision: 0.5877 - recall: 0.8489 - val_loss: 0.6664 - val_tp: 77.0000 - val_fp: 436.0000 - val_tn: 564.0000 - val_fn: 23.0000 - val_accuracy: 0.5827 - val_precision: 0.1501 - val_recall: 0.7700
    Epoch 9/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.8259 - tp: 470.6094 - fp: 293.1719 - tn: 205.8750 - fn: 69.3438 - accuracy: 0.6562 - precision: 0.6280 - recall: 0.8534 - val_loss: 0.6462 - val_tp: 76.0000 - val_fp: 381.0000 - val_tn: 619.0000 - val_fn: 24.0000 - val_accuracy: 0.6318 - val_precision: 0.1663 - val_recall: 0.7600
    Epoch 10/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.7778 - tp: 477.8125 - fp: 290.4688 - tn: 208.5781 - fn: 62.1406 - accuracy: 0.6680 - precision: 0.6400 - recall: 0.8570 - val_loss: 0.5779 - val_tp: 67.0000 - val_fp: 251.0000 - val_tn: 749.0000 - val_fn: 33.0000 - val_accuracy: 0.7418 - val_precision: 0.2107 - val_recall: 0.6700
    Epoch 11/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.7429 - tp: 460.7188 - fp: 241.9062 - tn: 257.1406 - fn: 79.2344 - accuracy: 0.6976 - precision: 0.6787 - recall: 0.8228 - val_loss: 0.5548 - val_tp: 60.0000 - val_fp: 255.0000 - val_tn: 745.0000 - val_fn: 40.0000 - val_accuracy: 0.7318 - val_precision: 0.1905 - val_recall: 0.6000
    Epoch 12/20
    63/63 [==============================] - 5s 87ms/step - loss: 0.7060 - tp: 472.8281 - fp: 207.7344 - tn: 291.3125 - fn: 67.1250 - accuracy: 0.7444 - precision: 0.7221 - recall: 0.8501 - val_loss: 0.5597 - val_tp: 62.0000 - val_fp: 259.0000 - val_tn: 741.0000 - val_fn: 38.0000 - val_accuracy: 0.7300 - val_precision: 0.1931 - val_recall: 0.6200
    Epoch 13/20
    63/63 [==============================] - 6s 88ms/step - loss: 0.6319 - tp: 483.8750 - fp: 186.8750 - tn: 312.1719 - fn: 56.0781 - accuracy: 0.7717 - precision: 0.7398 - recall: 0.8774 - val_loss: 0.4675 - val_tp: 53.0000 - val_fp: 175.0000 - val_tn: 825.0000 - val_fn: 47.0000 - val_accuracy: 0.7982 - val_precision: 0.2325 - val_recall: 0.5300
    Epoch 14/20
    63/63 [==============================] - 5s 85ms/step - loss: 0.6147 - tp: 472.5938 - fp: 171.8594 - tn: 327.1875 - fn: 67.3594 - accuracy: 0.7694 - precision: 0.7564 - recall: 0.8399 - val_loss: 0.5885 - val_tp: 72.0000 - val_fp: 292.0000 - val_tn: 708.0000 - val_fn: 28.0000 - val_accuracy: 0.7091 - val_precision: 0.1978 - val_recall: 0.7200
    Epoch 15/20
    63/63 [==============================] - 5s 86ms/step - loss: 0.5088 - tp: 492.9375 - fp: 136.9688 - tn: 362.0781 - fn: 47.0156 - accuracy: 0.8338 - precision: 0.8033 - recall: 0.9079 - val_loss: 0.5617 - val_tp: 75.0000 - val_fp: 257.0000 - val_tn: 743.0000 - val_fn: 25.0000 - val_accuracy: 0.7436 - val_precision: 0.2259 - val_recall: 0.7500
    Epoch 16/20
    63/63 [==============================] - 5s 85ms/step - loss: 0.4512 - tp: 499.7031 - fp: 123.9219 - tn: 375.1250 - fn: 40.2500 - accuracy: 0.8464 - precision: 0.8095 - recall: 0.9250 - val_loss: 0.4344 - val_tp: 65.0000 - val_fp: 162.0000 - val_tn: 838.0000 - val_fn: 35.0000 - val_accuracy: 0.8209 - val_precision: 0.2863 - val_recall: 0.6500
    Epoch 17/20
    63/63 [==============================] - 5s 85ms/step - loss: 0.4339 - tp: 502.6719 - fp: 110.8906 - tn: 388.1562 - fn: 37.2812 - accuracy: 0.8627 - precision: 0.8395 - recall: 0.9174 - val_loss: 0.5081 - val_tp: 68.0000 - val_fp: 207.0000 - val_tn: 793.0000 - val_fn: 32.0000 - val_accuracy: 0.7827 - val_precision: 0.2473 - val_recall: 0.6800
    Epoch 18/20
    63/63 [==============================] - 5s 85ms/step - loss: 0.3737 - tp: 506.1719 - fp: 87.2812 - tn: 411.7656 - fn: 33.7812 - accuracy: 0.8854 - precision: 0.8629 - recall: 0.9300 - val_loss: 0.5626 - val_tp: 72.0000 - val_fp: 225.0000 - val_tn: 775.0000 - val_fn: 28.0000 - val_accuracy: 0.7700 - val_precision: 0.2424 - val_recall: 0.7200
    Epoch 19/20
    63/63 [==============================] - 5s 85ms/step - loss: 0.3154 - tp: 511.9844 - fp: 81.8906 - tn: 417.1562 - fn: 27.9688 - accuracy: 0.8993 - precision: 0.8761 - recall: 0.9425 - val_loss: 0.7042 - val_tp: 80.0000 - val_fp: 323.0000 - val_tn: 677.0000 - val_fn: 20.0000 - val_accuracy: 0.6882 - val_precision: 0.1985 - val_recall: 0.8000
    Epoch 20/20
    63/63 [==============================] - 5s 85ms/step - loss: 0.3186 - tp: 510.7188 - fp: 79.2656 - tn: 419.7812 - fn: 29.2344 - accuracy: 0.9045 - precision: 0.8764 - recall: 0.9519 - val_loss: 0.6244 - val_tp: 72.0000 - val_fp: 249.0000 - val_tn: 751.0000 - val_fn: 28.0000 - val_accuracy: 0.7482 - val_precision: 0.2243 - val_recall: 0.7200
    

## Evaluation

#### Balanced Dataset


```python
plot_loss(history_cat, "Cat Training")
```


    
![png](2021-05-02-eval-on-unbalanced-datasets_files/2021-05-02-eval-on-unbalanced-datasets_40_0.png)
    



```python
plot_loss(history_dog, "Dog Training")
```


    
![png](2021-05-02-eval-on-unbalanced-datasets_files/2021-05-02-eval-on-unbalanced-datasets_41_0.png)
    



```python
eval_cat_balanced = model_cat.evaluate(test_ds_balanced, batch_size=BATCH_SIZE, verbose=1)
eval_dog_balanced = model_dog.evaluate(test_ds_balanced, batch_size=BATCH_SIZE, verbose=1)
```

    63/63 [==============================] - 4s 55ms/step - loss: 0.9864 - tp: 527.0000 - fp: 127.0000 - tn: 873.0000 - fn: 473.0000 - accuracy: 0.7000 - precision: 0.8058 - recall: 0.5270
    63/63 [==============================] - 4s 55ms/step - loss: 0.7060 - tp: 687.0000 - fp: 240.0000 - tn: 760.0000 - fn: 313.0000 - accuracy: 0.7235 - precision: 0.7411 - recall: 0.6870
    


```python
for name, value in zip(model_cat.metrics_names, eval_cat_balanced):
    print(name, ': ', value)
```

    loss :  0.9864482879638672
    tp :  527.0
    fp :  127.0
    tn :  873.0
    fn :  473.0
    accuracy :  0.699999988079071
    precision :  0.8058103919029236
    recall :  0.5270000100135803
    


```python
for name, value in zip(model_dog.metrics_names, eval_dog_balanced):
    print(name, ': ', value)
```

    loss :  0.7059767842292786
    tp :  687.0
    fp :  240.0
    tn :  760.0
    fn :  313.0
    accuracy :  0.7235000133514404
    precision :  0.7411003112792969
    recall :  0.6869999766349792
    

In the balanced one, the dog model has higher recall but lower precision. It also has higher accuracy, but that's much closer between the models. Let's look at the F1 scores.


```python
cat_f1_balanced = calc_f1(eval_cat_balanced)
dog_f1_balanced = calc_f1(eval_dog_balanced)
print(f"Cat model F1 score: {round(cat_f1_balanced, 4)}")
print(f"Dog model F1 score: {round(dog_f1_balanced, 4)}")
```

    Cat model F1 score: 0.6372
    Dog model F1 score: 0.713
    

The dog model has a higher F1 score. Now let's look at the confusion matrices.


```python
cat_preds_balanced = model_cat.predict(test_ds_balanced)
dog_preds_balanced = model_dog.predict(test_ds_balanced)
true_labels_balanced = tf.concat([y for x, y in test_ds_balanced], axis=0)
```


```python
plot_cm(true_labels_balanced, cat_preds_balanced)
```


    
![png](2021-05-02-eval-on-unbalanced-datasets_files/2021-05-02-eval-on-unbalanced-datasets_49_0.png)
    



```python
plot_cm(true_labels_balanced, dog_preds_balanced)
```


    
![png](2021-05-02-eval-on-unbalanced-datasets_files/2021-05-02-eval-on-unbalanced-datasets_50_0.png)
    


We can see that the dog model predicted more dogs and the cat model more cats, as expected.

#### Unbalanced Dataset


```python
eval_cat_unbalanced = model_cat.evaluate(test_ds_unbalanced, batch_size=BATCH_SIZE, verbose=1)
eval_dog_unbalanced = model_dog.evaluate(test_ds_unbalanced, batch_size=BATCH_SIZE, verbose=1)
```

    35/35 [==============================] - 2s 55ms/step - loss: 0.4402 - tp: 50.0000 - fp: 141.0000 - tn: 859.0000 - fn: 50.0000 - accuracy: 0.8264 - precision: 0.2618 - recall: 0.5000
    35/35 [==============================] - 2s 54ms/step - loss: 0.6637 - tp: 59.0000 - fp: 230.0000 - tn: 770.0000 - fn: 41.0000 - accuracy: 0.7536 - precision: 0.2042 - recall: 0.5900
    


```python
for name, value in zip(model_cat.metrics_names, eval_cat_unbalanced):
    print(name, ': ', value)
```

    loss :  0.44017571210861206
    tp :  50.0
    fp :  141.0
    tn :  859.0
    fn :  50.0
    accuracy :  0.8263636231422424
    precision :  0.26178011298179626
    recall :  0.5
    


```python
for name, value in zip(model_dog.metrics_names, eval_dog_unbalanced):
    print(name, ': ', value)
```

    loss :  0.6636688709259033
    tp :  59.0
    fp :  230.0
    tn :  770.0
    fn :  41.0
    accuracy :  0.753636360168457
    precision :  0.20415225625038147
    recall :  0.5899999737739563
    

In the unbalanced dataset, the dog model has higher recall but lower precision and this time much lower accuracy.


```python
cat_f1_unbalanced = calc_f1(eval_cat_unbalanced)
dog_f1_unbalanced = calc_f1(eval_dog_unbalanced)
print(f"Cat model F1 score: {round(cat_f1_unbalanced, 4)}")
print(f"Dog model F1 score: {round(dog_f1_unbalanced, 4)}")
```

    Cat model F1 score: 0.3436
    Dog model F1 score: 0.3033
    

## Conclusion

Immediately, we see that the performance of *both* models is far worse on the unbalanced dataset. But that's how the real world data is going to be (in our pretend universe), so those are the metrics we need. Even more interesting, we see that while the **dog** model had the better score on the balanced test set, the **cat** model had the better one on the unbalanced dataset. However, the scores are quite close and given the dataset size, there's a lot of uncertainty associated with these measurements. I would say only that this at least _shows_ that this can happen.


```python

```
