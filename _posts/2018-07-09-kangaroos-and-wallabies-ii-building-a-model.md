---
layout: post
title: "Kangaroos and Wallabies II: Building a Model"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/ki_kangs.jpg"
tags: [Computer Vision, Convolutional Neural Networks, Python, TensorFlow, Wildlife]
---

In this notebook, we're going to take the [datatset we prepared](https://jss367.github.io/kangaroos-and-wallabies-i-preparing-the-data.html) and build a model to classify the images.

<b>Table of contents</b>
* TOC
{:toc}

## Introduction

What model should we use? A simple model, such as logistic regression, is able to do fairly well on a simple image classification task like [MNIST](http://yann.lecun.com/exdb/mnist/) or even cat vs non-cat, but won't really cut it for this task. Because kangaroos and wallabies are so similar, even a fully connected neural network doesn't score particularly well on this dataset.

So we're just going to skip those and go straight to the most popular model in modern computer vision: convolutional neural networks. We're going to take a huge model known as [Xception](https://arxiv.org/abs/1610.02357) and apply it to our dataset. But we're not just going to take the model architecture, we're going to take the actual ImageNet weights that have been developed by weeks of training it on the ImageNet dataset. Then we will freeze everything except the last layers and train them with our dataset. This is known as transfer learning and it allows us to benefit from the highly tuned convolutional layers that are so good at extracting features while allowing us to tweak the last layers specifically for our problem.


```python
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
```

Let's make sure we have a GPU available because this would take way too long on a CPU.


```python
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```

    Num GPUs Available:  1


## Prepare the Data

Now we'll prepared our data using TensorFlow's `tf.data.Datasets`.


```python
train_data_dir = Path('E:/WallabiesAndRoosFullSize/train')
val_data_dir = Path('E:/WallabiesAndRoosFullSize/val')
test_data_dir = Path('E:/WallabiesAndRoosFullSize/test')
```


```python
train_files = tf.data.Dataset.list_files(str(train_data_dir/'*/*'), shuffle=True)
val_files = tf.data.Dataset.list_files(str(val_data_dir/'*/*'), shuffle=False)
test_files = tf.data.Dataset.list_files(str(test_data_dir/'*/*'), shuffle=False)
```


```python
class_names = np.array(sorted([folder.name for folder in train_data_dir.glob('*')]))
num_classes = len(class_names)
print(class_names)
```

    ['kangaroo' 'wallaby']


We'll set the image size we want to work with and the number of epochs.


```python
img_height, img_width = 256, 256
BATCH_SIZE = 4
EPOCHS = 5
```


```python
def parse_label(filename):
    parts = tf.strings.split(filename, sep=os.path.sep)
    one_hot_label = parts[-2] == class_names
    label = tf.argmax(one_hot_label)
    return label

def parse_image(filename):
    img = tf.io.read_file(filename)
    image_decoded = tf.io.decode_jpeg(img, channels=3)
    image = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image = tf.image.resize(image, (img_height, img_width))
    return image

def parse_file_path(filename):
    image = parse_image(filename)
    label = parse_label(filename)
    return image, label

```

Now let's look at some of our images


```python
data_iter = iter(train_files)
file_path = next(data_iter)
image, label = parse_file_path(file_path)

def display_image(image, label):
    plt.figure()
    plt.imshow(image)
    plt.title(class_names[label])
    plt.axis('off')

display_image(image, label)
```


    
![png](2018-07-09-kangaroos-and-wallabies-II-building-a-model_files/2018-07-09-kangaroos-and-wallabies-II-building-a-model_18_0.png)
    



```python
file_path = next(data_iter)
image, label = parse_file_path(file_path)
display_image(image, label)
```


    
![png](2018-07-09-kangaroos-and-wallabies-II-building-a-model_files/2018-07-09-kangaroos-and-wallabies-II-building-a-model_19_0.png)
    


Now we create the image data pipeline. This is the process of steps to take us from a `tf.data.Dataset` object into parsed and batched image and label pairs ready for training. We'll also use TensorFlow's `prefetch` ability to speed up the data loading.


```python
def one_hot(image, label):
    """
    Convert to one-hot encoding
    """
    label = tf.one_hot(tf.cast(label, tf.int32), num_classes)
    # Recasts it to Float32
    label = tf.cast(label, tf.float32)
    return image, label
```


```python
train_dataset = train_files.map(parse_file_path)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.map(one_hot)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
```


```python
val_dataset = val_files.map(parse_file_path)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.map(one_hot)
val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
```


```python
test_dataset = test_files.map(parse_file_path)
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(one_hot)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
```


```python
data_iter = iter(train_dataset)
examp_im, examp_label = next(data_iter)
```

Let's visualize the data coming out of our `tf.data.Dataset`. I like to do this to ensure it's what I'm expecting. The image should be a bunch of values between 0 and 1 and the labels should all be either 0 or 1.


```python
plt.hist(examp_im.numpy().flatten());
```


    
![png](2018-07-09-kangaroos-and-wallabies-II-building-a-model_files/2018-07-09-kangaroos-and-wallabies-II-building-a-model_27_0.png)
    



```python
np.unique(examp_label)
```




    array([0., 1.], dtype=float32)



Looks good!

## Establish a Baseline

Before we build a complex model, we should develop a baseline so we can gauge our model's performance. As we saw in the [first post](https://jss367.github.io/Preparing-folder-structure.html), there are more images of kangaroos than of wallabies, so we need to take that into consideration. For example, if our model accurately predicted 75% of images, that might sound good. But what if simply guessing "kangaroo" each time got the same result, then maybe the model isn't learning much at all. So what's the baseline? How many would we get from pure guessing? To answer that, we'll use a dummy classifier. Our dummy classifier will find the most common label and predict that for every image.

If we had many classes, we could use something like sklearn's [DummyClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html) with the `most_frequent` strategy, but we only have two classes so we can simplify the process. Let's figure out which class has more examples and then guess that for every example in our test dataset.

We know that all of our classes are either 0 or 1, so we can see which we have more of like so.


```python
kang_dir = test_data_dir / 'kangaroo'
wall_dir = test_data_dir / 'wallaby'
```


```python
num_kang_images = len(list(kang_dir.glob('*')))
num_wall_images = len(list(wall_dir.glob('*')))
```


```python
print(f"Number of kangaroo images: {num_kang_images}")
print(f"Number of wallaby images: {num_wall_images}")
print(f"If we just guessed the most common class each time, we would be right {num_kang_images / (num_kang_images + num_wall_images):.2%} of the time.")
```

    Number of kangaroo images: 306
    Number of wallaby images: 195
    If we just guessed the most common class each time, we would be right 61.08% of the time.


## Prepare the Model

We have to provide some basic characteristics of the network and how we want to train it. As we said before, we'll use ImageNet weights.


```python
model_encoder = applications.Xception(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
```


```python
# Add our own layers at the end 
x = model_encoder.output
x = Flatten()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = Dense(256,
          activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)
```

Create the model using the inputs from the pretrained model.


```python
model = Model(inputs = model_encoder.input, outputs = predictions)
```

We will freeze all layers except the last five. Then we'll add our own layers at the end. For the final activation function, I'm going to use [softmax](https://en.wikipedia.org/wiki/Softmax_function). I could just use a sigmoid function, but using softmax makes it easier to scale the model to multiple classes, even though they're mathematically equivalent in the case of two classes.

One problem we're sure to run into with such a large and complex neural network is overfitting. This is when the model finds a small number of features that work well in the dataset but might not generalize to all images. For example, say all the images in the training set show the ears of the kangaroos and wallabies really well and the model learns how to distinguish them based on that. Well, maybe the ears are behind a branch in some other images, then how is the model going to decide? We want to model to look at many aspects of the image and use them all to classify it.

There are several different types of regularization that we could use, but we'll discuss just two of them: L1 and L2. They rely on the same concept - penalizing the network for weights that are too large. This forces each individual parameter to be low and therefore prevents the model from relying too much on a single weight or feature. L1 regularization penalizes based on the magnitude of the weights, and L2 penalizes based on the <i>square</i> of the magnitude of the weights. There are good reasons to use one over the other which we won't get into, but in this case, we'll use L2 because it's more common and generally seems to give better performance in most cases.


```python
for layer in model.layers[:-2]:
    layer.trainable = False
```


```python
model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 256, 256, 3) 0                                            
    __________________________________________________________________________________________________
    block1_conv1 (Conv2D)           (None, 127, 127, 32) 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    block1_conv1_bn (BatchNormaliza (None, 127, 127, 32) 128         block1_conv1[0][0]               
    __________________________________________________________________________________________________
    block1_conv1_act (Activation)   (None, 127, 127, 32) 0           block1_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    block1_conv2 (Conv2D)           (None, 125, 125, 64) 18432       block1_conv1_act[0][0]           
    __________________________________________________________________________________________________
    block1_conv2_bn (BatchNormaliza (None, 125, 125, 64) 256         block1_conv2[0][0]               
    __________________________________________________________________________________________________
    block1_conv2_act (Activation)   (None, 125, 125, 64) 0           block1_conv2_bn[0][0]            
    __________________________________________________________________________________________________
    block2_sepconv1 (SeparableConv2 (None, 125, 125, 128 8768        block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_sepconv1_bn (BatchNormal (None, 125, 125, 128 512         block2_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block2_sepconv2_act (Activation (None, 125, 125, 128 0           block2_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block2_sepconv2 (SeparableConv2 (None, 125, 125, 128 17536       block2_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block2_sepconv2_bn (BatchNormal (None, 125, 125, 128 512         block2_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 63, 63, 128)  8192        block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_pool (MaxPooling2D)      (None, 63, 63, 128)  0           block2_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 63, 63, 128)  512         conv2d[0][0]                     
    __________________________________________________________________________________________________
    add (Add)                       (None, 63, 63, 128)  0           block2_pool[0][0]                
                                                                     batch_normalization[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv1_act (Activation (None, 63, 63, 128)  0           add[0][0]                        
    __________________________________________________________________________________________________
    block3_sepconv1 (SeparableConv2 (None, 63, 63, 256)  33920       block3_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv1_bn (BatchNormal (None, 63, 63, 256)  1024        block3_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block3_sepconv2_act (Activation (None, 63, 63, 256)  0           block3_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block3_sepconv2 (SeparableConv2 (None, 63, 63, 256)  67840       block3_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv2_bn (BatchNormal (None, 63, 63, 256)  1024        block3_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 32, 32, 256)  32768       add[0][0]                        
    __________________________________________________________________________________________________
    block3_pool (MaxPooling2D)      (None, 32, 32, 256)  0           block3_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 32, 32, 256)  1024        conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 32, 32, 256)  0           block3_pool[0][0]                
                                                                     batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    block4_sepconv1_act (Activation (None, 32, 32, 256)  0           add_1[0][0]                      
    __________________________________________________________________________________________________
    block4_sepconv1 (SeparableConv2 (None, 32, 32, 728)  188672      block4_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block4_sepconv1_bn (BatchNormal (None, 32, 32, 728)  2912        block4_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block4_sepconv2_act (Activation (None, 32, 32, 728)  0           block4_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block4_sepconv2 (SeparableConv2 (None, 32, 32, 728)  536536      block4_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block4_sepconv2_bn (BatchNormal (None, 32, 32, 728)  2912        block4_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 16, 16, 728)  186368      add_1[0][0]                      
    __________________________________________________________________________________________________
    block4_pool (MaxPooling2D)      (None, 16, 16, 728)  0           block4_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 16, 16, 728)  2912        conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 16, 16, 728)  0           block4_pool[0][0]                
                                                                     batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    block5_sepconv1_act (Activation (None, 16, 16, 728)  0           add_2[0][0]                      
    __________________________________________________________________________________________________
    block5_sepconv1 (SeparableConv2 (None, 16, 16, 728)  536536      block5_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv1_bn (BatchNormal (None, 16, 16, 728)  2912        block5_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block5_sepconv2_act (Activation (None, 16, 16, 728)  0           block5_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block5_sepconv2 (SeparableConv2 (None, 16, 16, 728)  536536      block5_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv2_bn (BatchNormal (None, 16, 16, 728)  2912        block5_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block5_sepconv3_act (Activation (None, 16, 16, 728)  0           block5_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block5_sepconv3 (SeparableConv2 (None, 16, 16, 728)  536536      block5_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv3_bn (BatchNormal (None, 16, 16, 728)  2912        block5_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 16, 16, 728)  0           block5_sepconv3_bn[0][0]         
                                                                     add_2[0][0]                      
    __________________________________________________________________________________________________
    block6_sepconv1_act (Activation (None, 16, 16, 728)  0           add_3[0][0]                      
    __________________________________________________________________________________________________
    block6_sepconv1 (SeparableConv2 (None, 16, 16, 728)  536536      block6_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv1_bn (BatchNormal (None, 16, 16, 728)  2912        block6_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block6_sepconv2_act (Activation (None, 16, 16, 728)  0           block6_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block6_sepconv2 (SeparableConv2 (None, 16, 16, 728)  536536      block6_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv2_bn (BatchNormal (None, 16, 16, 728)  2912        block6_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block6_sepconv3_act (Activation (None, 16, 16, 728)  0           block6_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block6_sepconv3 (SeparableConv2 (None, 16, 16, 728)  536536      block6_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv3_bn (BatchNormal (None, 16, 16, 728)  2912        block6_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, 16, 16, 728)  0           block6_sepconv3_bn[0][0]         
                                                                     add_3[0][0]                      
    __________________________________________________________________________________________________
    block7_sepconv1_act (Activation (None, 16, 16, 728)  0           add_4[0][0]                      
    __________________________________________________________________________________________________
    block7_sepconv1 (SeparableConv2 (None, 16, 16, 728)  536536      block7_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv1_bn (BatchNormal (None, 16, 16, 728)  2912        block7_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block7_sepconv2_act (Activation (None, 16, 16, 728)  0           block7_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block7_sepconv2 (SeparableConv2 (None, 16, 16, 728)  536536      block7_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv2_bn (BatchNormal (None, 16, 16, 728)  2912        block7_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block7_sepconv3_act (Activation (None, 16, 16, 728)  0           block7_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block7_sepconv3 (SeparableConv2 (None, 16, 16, 728)  536536      block7_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv3_bn (BatchNormal (None, 16, 16, 728)  2912        block7_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, 16, 16, 728)  0           block7_sepconv3_bn[0][0]         
                                                                     add_4[0][0]                      
    __________________________________________________________________________________________________
    block8_sepconv1_act (Activation (None, 16, 16, 728)  0           add_5[0][0]                      
    __________________________________________________________________________________________________
    block8_sepconv1 (SeparableConv2 (None, 16, 16, 728)  536536      block8_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv1_bn (BatchNormal (None, 16, 16, 728)  2912        block8_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block8_sepconv2_act (Activation (None, 16, 16, 728)  0           block8_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block8_sepconv2 (SeparableConv2 (None, 16, 16, 728)  536536      block8_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv2_bn (BatchNormal (None, 16, 16, 728)  2912        block8_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block8_sepconv3_act (Activation (None, 16, 16, 728)  0           block8_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block8_sepconv3 (SeparableConv2 (None, 16, 16, 728)  536536      block8_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv3_bn (BatchNormal (None, 16, 16, 728)  2912        block8_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, 16, 16, 728)  0           block8_sepconv3_bn[0][0]         
                                                                     add_5[0][0]                      
    __________________________________________________________________________________________________
    block9_sepconv1_act (Activation (None, 16, 16, 728)  0           add_6[0][0]                      
    __________________________________________________________________________________________________
    block9_sepconv1 (SeparableConv2 (None, 16, 16, 728)  536536      block9_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv1_bn (BatchNormal (None, 16, 16, 728)  2912        block9_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block9_sepconv2_act (Activation (None, 16, 16, 728)  0           block9_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block9_sepconv2 (SeparableConv2 (None, 16, 16, 728)  536536      block9_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv2_bn (BatchNormal (None, 16, 16, 728)  2912        block9_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block9_sepconv3_act (Activation (None, 16, 16, 728)  0           block9_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block9_sepconv3 (SeparableConv2 (None, 16, 16, 728)  536536      block9_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv3_bn (BatchNormal (None, 16, 16, 728)  2912        block9_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, 16, 16, 728)  0           block9_sepconv3_bn[0][0]         
                                                                     add_6[0][0]                      
    __________________________________________________________________________________________________
    block10_sepconv1_act (Activatio (None, 16, 16, 728)  0           add_7[0][0]                      
    __________________________________________________________________________________________________
    block10_sepconv1 (SeparableConv (None, 16, 16, 728)  536536      block10_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv1_bn (BatchNorma (None, 16, 16, 728)  2912        block10_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block10_sepconv2_act (Activatio (None, 16, 16, 728)  0           block10_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block10_sepconv2 (SeparableConv (None, 16, 16, 728)  536536      block10_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv2_bn (BatchNorma (None, 16, 16, 728)  2912        block10_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block10_sepconv3_act (Activatio (None, 16, 16, 728)  0           block10_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block10_sepconv3 (SeparableConv (None, 16, 16, 728)  536536      block10_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv3_bn (BatchNorma (None, 16, 16, 728)  2912        block10_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, 16, 16, 728)  0           block10_sepconv3_bn[0][0]        
                                                                     add_7[0][0]                      
    __________________________________________________________________________________________________
    block11_sepconv1_act (Activatio (None, 16, 16, 728)  0           add_8[0][0]                      
    __________________________________________________________________________________________________
    block11_sepconv1 (SeparableConv (None, 16, 16, 728)  536536      block11_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv1_bn (BatchNorma (None, 16, 16, 728)  2912        block11_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block11_sepconv2_act (Activatio (None, 16, 16, 728)  0           block11_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block11_sepconv2 (SeparableConv (None, 16, 16, 728)  536536      block11_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv2_bn (BatchNorma (None, 16, 16, 728)  2912        block11_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block11_sepconv3_act (Activatio (None, 16, 16, 728)  0           block11_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block11_sepconv3 (SeparableConv (None, 16, 16, 728)  536536      block11_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv3_bn (BatchNorma (None, 16, 16, 728)  2912        block11_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, 16, 16, 728)  0           block11_sepconv3_bn[0][0]        
                                                                     add_8[0][0]                      
    __________________________________________________________________________________________________
    block12_sepconv1_act (Activatio (None, 16, 16, 728)  0           add_9[0][0]                      
    __________________________________________________________________________________________________
    block12_sepconv1 (SeparableConv (None, 16, 16, 728)  536536      block12_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv1_bn (BatchNorma (None, 16, 16, 728)  2912        block12_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block12_sepconv2_act (Activatio (None, 16, 16, 728)  0           block12_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block12_sepconv2 (SeparableConv (None, 16, 16, 728)  536536      block12_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv2_bn (BatchNorma (None, 16, 16, 728)  2912        block12_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block12_sepconv3_act (Activatio (None, 16, 16, 728)  0           block12_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block12_sepconv3 (SeparableConv (None, 16, 16, 728)  536536      block12_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv3_bn (BatchNorma (None, 16, 16, 728)  2912        block12_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, 16, 16, 728)  0           block12_sepconv3_bn[0][0]        
                                                                     add_9[0][0]                      
    __________________________________________________________________________________________________
    block13_sepconv1_act (Activatio (None, 16, 16, 728)  0           add_10[0][0]                     
    __________________________________________________________________________________________________
    block13_sepconv1 (SeparableConv (None, 16, 16, 728)  536536      block13_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block13_sepconv1_bn (BatchNorma (None, 16, 16, 728)  2912        block13_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block13_sepconv2_act (Activatio (None, 16, 16, 728)  0           block13_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block13_sepconv2 (SeparableConv (None, 16, 16, 1024) 752024      block13_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block13_sepconv2_bn (BatchNorma (None, 16, 16, 1024) 4096        block13_sepconv2[0][0]           
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 8, 8, 1024)   745472      add_10[0][0]                     
    __________________________________________________________________________________________________
    block13_pool (MaxPooling2D)     (None, 8, 8, 1024)   0           block13_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 8, 8, 1024)   4096        conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, 8, 8, 1024)   0           block13_pool[0][0]               
                                                                     batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    block14_sepconv1 (SeparableConv (None, 8, 8, 1536)   1582080     add_11[0][0]                     
    __________________________________________________________________________________________________
    block14_sepconv1_bn (BatchNorma (None, 8, 8, 1536)   6144        block14_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block14_sepconv1_act (Activatio (None, 8, 8, 1536)   0           block14_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block14_sepconv2 (SeparableConv (None, 8, 8, 2048)   3159552     block14_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block14_sepconv2_bn (BatchNorma (None, 8, 8, 2048)   8192        block14_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block14_sepconv2_act (Activatio (None, 8, 8, 2048)   0           block14_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 131072)       0           block14_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    batch_normalization_4 (BatchNor (None, 131072)       524288      flatten[0][0]                    
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 256)          33554688    batch_normalization_4[0][0]      
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 2)            514         dense[0][0]                      
    ==================================================================================================
    Total params: 54,940,970
    Trainable params: 33,555,202
    Non-trainable params: 21,385,768
    __________________________________________________________________________________________________


## Compile the Model

Now we compile the model. The compiler automatically determines how to split up the data between the CPU and GPU. We have to specify the loss function, optimizer, and whatever metrics we're interested in.


```python
model.compile(loss="binary_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
```

I always like to save the model so I can reload it later if needed.


```python
results_path = Path(r'C:\Users\Julius\Documents\GitHub\cv\results\KangWall_' + datetime.now().strftime("%Y%m%d-%H%M%S"))

os.makedirs(results_path, exist_ok=True)
```


```python
# Save the model according to the conditions

checkpoint_path = results_path / "model-{epoch:02d}-{val_accuracy:.2f}.hdf5"
early = EarlyStopping(monitor='val_accuracy', min_delta=0,
                      patience=4, verbose=1, mode='auto')
checkpoint = ModelCheckpoint(str(checkpoint_path), monitor='val_accuracy', verbose=1, save_best_only=False, mode='auto', save_freq='epoch')
```


```python
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=results_path, histogram_freq=0, write_graph=True, write_images=True,
    update_freq='epoch')
```

OK, now let's train the model.


```python
history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[tensorboard_callback, early, checkpoint], 
                validation_data=val_dataset)
```

    Epoch 1/5
    914/914 [==============================] - 735s 796ms/step - loss: 0.2741 - accuracy: 0.8859 - val_loss: 0.5043 - val_accuracy: 0.8448
    
    Epoch 00001: saving model to C:\Users\Julius\Documents\GitHub\cv\results\KangWall_20210708-191155\model-01-0.84.hdf5
    Epoch 2/5
    914/914 [==============================] - 612s 670ms/step - loss: 0.0601 - accuracy: 0.9840 - val_loss: 0.5147 - val_accuracy: 0.8325
    
    Epoch 00002: saving model to C:\Users\Julius\Documents\GitHub\cv\results\KangWall_20210708-191155\model-02-0.83.hdf5
    Epoch 3/5
    914/914 [==============================] - 549s 601ms/step - loss: 0.0306 - accuracy: 0.9958 - val_loss: 0.5672 - val_accuracy: 0.8377
    
    Epoch 00003: saving model to C:\Users\Julius\Documents\GitHub\cv\results\KangWall_20210708-191155\model-03-0.84.hdf5
    Epoch 4/5
    914/914 [==============================] - 549s 601ms/step - loss: 0.0235 - accuracy: 0.9968 - val_loss: 0.5748 - val_accuracy: 0.8377
    
    Epoch 00004: saving model to C:\Users\Julius\Documents\GitHub\cv\results\KangWall_20210708-191155\model-04-0.84.hdf5
    Epoch 5/5
    914/914 [==============================] - 551s 603ms/step - loss: 0.0121 - accuracy: 0.9993 - val_loss: 0.5871 - val_accuracy: 0.8377
    
    Epoch 00005: saving model to C:\Users\Julius\Documents\GitHub\cv\results\KangWall_20210708-191155\model-05-0.84.hdf5
    Epoch 00005: early stopping



```python
model.save(str(results_path) + 'final_model')
```

    INFO:tensorflow:Assets written to: C:\Users\Julius\Documents\GitHub\cv\results\KangWall_20210708-191155final_model\assets


Note how much higher the training accuracy is than the validation accuracy. That means we're overfitting the training data. We'll go over how to correct for that in a future notebook.


```python
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
```




    <matplotlib.legend.Legend at 0x1d210caab80>




    
![png](2018-07-09-kangaroos-and-wallabies-II-building-a-model_files/2018-07-09-kangaroos-and-wallabies-II-building-a-model_57_1.png)
    


## Evaluating the Model

Note that we're only saving the model when it improves the validation set. Since we keep checking the accuracy on our validation set, we could actually be overfitting the validation set as well. That's why we reserved a test set that, so far, we haven't even looked at. We'll use that to compute the model's accuracy. Accuracy isn't the most comprehensive way to measure model quality, especially in the case of multiple classes, but it's quick and simple, so we'll use it.


```python
model_eval = tf.keras.models.load_model(str(results_path) + 'final_model')
```


```python
model_eval.evaluate(test_dataset)
```

    126/126 [==============================] - 88s 671ms/step - loss: 0.3407 - accuracy: 0.8922





    [0.34070613980293274, 0.8922155499458313]



89% - that's pretty good!. But we're not done yet. In the next notebook, we'll talk about data augmentation and how we can use that to improve the model's performance on the test set.
