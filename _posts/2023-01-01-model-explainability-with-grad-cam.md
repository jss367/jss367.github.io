---
layout: post
title: "Model Explainability with Grad-CAM"
description: "This post is a tutorial for how to use Grad-CAM to explain computer vision models."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/camel.jpg"
tags: [Model Explainability, Python, PyTorch]
---

This post is a tutorial of how to use Grad-CAM to explain a neural network outputs. [Grad-CAM](https://arxiv.org/abs/1610.02391) is a technique for visualizing the regions in an image that are most important for a convolutional neural network (CNN) to make a prediction. It can be used with any CNN, but it is most commonly used with image classification models. This tutorial uses some code from the [keras tutorial](https://keras.io/examples/vision/grad_cam/).


```python
import matplotlib.cm as cm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from pyxtend import struct
from tensorflow.keras.applications.xception import Xception, decode_predictions, preprocess_input
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
```

# Load the Image


```python
img_path = 'cat_and_dog_hats.png'
```


```python
display(Image(img_path))
```


    
![png](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_5_0.png)
    


This isn't a good image for single-label image classification because it has two different classes in it. But I'm going to use it anyway so we can look at how to focus on specific classes within an image.

# Create a Model


```python
model = Xception(weights="imagenet")
```

What we want is a model that returns the output of the last convolutional layer. To do this, we'll have to make some customizations to the model. We have to remove the softmax function in the last layer.


```python
# Remove last layer's softmax
model.layers[-1].activation = None
```

# Preprocess the Image

Now that we have a model we can preprocess the image for the model.


```python
img_size = (299, 299) # default for Xception
img = load_img(img_path, target_size=img_size)
array = img_to_array(img)
array = np.expand_dims(array, axis=0)
```


```python
img_array = preprocess_input(array)
```

# Predict the Class

Now we can make a prediction.


```python
preds = model.predict(img_array)
```


```python
struct(preds)
```




    {numpy.ndarray: [{numpy.ndarray: [numpy.float32,
        numpy.float32,
        numpy.float32,
        '...1000 total']}]}



The prediction is a numpy array of values. To understand the predictions, we'll have to use xception's `decode_predictions` function. Let's look at the top 10 predictions.


```python
preds = model.predict(img_array)
decode_predictions(preds, top=10)[0]
```

    Downloading data from https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json
    40960/35363 [==================================] - 0s 1us/step
    




    [('n02100583', 'vizsla', 6.225254),
     ('n02087394', 'Rhodesian_ridgeback', 6.069996),
     ('n02108089', 'boxer', 4.5338044),
     ('n02108422', 'bull_mastiff', 4.489726),
     ('n02090379', 'redbone', 4.272118),
     ('n02088466', 'bloodhound', 4.1386704),
     ('n02109047', 'Great_Dane', 3.9553747),
     ('n02099712', 'Labrador_retriever', 3.943748),
     ('n03724870', 'mask', 3.6265507),
     ('n03494278', 'harmonica', 3.284974)]



# Next Step

OK, now we have predictions. Now we have to create a model that outputs the activations of the last convolutional layer as well as the output predictions.

Ee don't know the name of the last convolutional layer, but we can print out all the layer names and look for the last one before a `flatten` or `avg_pool` layer. We know it's going to be one of the last layers, so we'll only print out the last ten.


```python
[l.name for l in model.layers[-10:]]
```




    ['batch_normalization_3',
     'add_11',
     'block14_sepconv1',
     'block14_sepconv1_bn',
     'block14_sepconv1_act',
     'block14_sepconv2',
     'block14_sepconv2_bn',
     'block14_sepconv2_act',
     'avg_pool',
     'predictions']



Nice! Looks like `block14_sepconv2_act` is what we're looking for.


```python
last_conv_layer_name = "block14_sepconv2_act"
```


```python
grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
```


```python
grad_model.summary()
```

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 299, 299, 3) 0                                            
    __________________________________________________________________________________________________
    block1_conv1 (Conv2D)           (None, 149, 149, 32) 864         input_1[0][0]                    
    __________________________________________________________________________________________________
    block1_conv1_bn (BatchNormaliza (None, 149, 149, 32) 128         block1_conv1[0][0]               
    __________________________________________________________________________________________________
    block1_conv1_act (Activation)   (None, 149, 149, 32) 0           block1_conv1_bn[0][0]            
    __________________________________________________________________________________________________
    block1_conv2 (Conv2D)           (None, 147, 147, 64) 18432       block1_conv1_act[0][0]           
    __________________________________________________________________________________________________
    block1_conv2_bn (BatchNormaliza (None, 147, 147, 64) 256         block1_conv2[0][0]               
    __________________________________________________________________________________________________
    block1_conv2_act (Activation)   (None, 147, 147, 64) 0           block1_conv2_bn[0][0]            
    __________________________________________________________________________________________________
    block2_sepconv1 (SeparableConv2 (None, 147, 147, 128 8768        block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_sepconv1_bn (BatchNormal (None, 147, 147, 128 512         block2_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block2_sepconv2_act (Activation (None, 147, 147, 128 0           block2_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block2_sepconv2 (SeparableConv2 (None, 147, 147, 128 17536       block2_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block2_sepconv2_bn (BatchNormal (None, 147, 147, 128 512         block2_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 74, 74, 128)  8192        block1_conv2_act[0][0]           
    __________________________________________________________________________________________________
    block2_pool (MaxPooling2D)      (None, 74, 74, 128)  0           block2_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization (BatchNorma (None, 74, 74, 128)  512         conv2d[0][0]                     
    __________________________________________________________________________________________________
    add (Add)                       (None, 74, 74, 128)  0           block2_pool[0][0]                
                                                                     batch_normalization[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv1_act (Activation (None, 74, 74, 128)  0           add[0][0]                        
    __________________________________________________________________________________________________
    block3_sepconv1 (SeparableConv2 (None, 74, 74, 256)  33920       block3_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv1_bn (BatchNormal (None, 74, 74, 256)  1024        block3_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block3_sepconv2_act (Activation (None, 74, 74, 256)  0           block3_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block3_sepconv2 (SeparableConv2 (None, 74, 74, 256)  67840       block3_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block3_sepconv2_bn (BatchNormal (None, 74, 74, 256)  1024        block3_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 37, 37, 256)  32768       add[0][0]                        
    __________________________________________________________________________________________________
    block3_pool (MaxPooling2D)      (None, 37, 37, 256)  0           block3_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_1 (BatchNor (None, 37, 37, 256)  1024        conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    add_1 (Add)                     (None, 37, 37, 256)  0           block3_pool[0][0]                
                                                                     batch_normalization_1[0][0]      
    __________________________________________________________________________________________________
    block4_sepconv1_act (Activation (None, 37, 37, 256)  0           add_1[0][0]                      
    __________________________________________________________________________________________________
    block4_sepconv1 (SeparableConv2 (None, 37, 37, 728)  188672      block4_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block4_sepconv1_bn (BatchNormal (None, 37, 37, 728)  2912        block4_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block4_sepconv2_act (Activation (None, 37, 37, 728)  0           block4_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block4_sepconv2 (SeparableConv2 (None, 37, 37, 728)  536536      block4_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block4_sepconv2_bn (BatchNormal (None, 37, 37, 728)  2912        block4_sepconv2[0][0]            
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 19, 19, 728)  186368      add_1[0][0]                      
    __________________________________________________________________________________________________
    block4_pool (MaxPooling2D)      (None, 19, 19, 728)  0           block4_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    batch_normalization_2 (BatchNor (None, 19, 19, 728)  2912        conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    add_2 (Add)                     (None, 19, 19, 728)  0           block4_pool[0][0]                
                                                                     batch_normalization_2[0][0]      
    __________________________________________________________________________________________________
    block5_sepconv1_act (Activation (None, 19, 19, 728)  0           add_2[0][0]                      
    __________________________________________________________________________________________________
    block5_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block5_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block5_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block5_sepconv2_act (Activation (None, 19, 19, 728)  0           block5_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block5_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block5_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block5_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block5_sepconv3_act (Activation (None, 19, 19, 728)  0           block5_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block5_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block5_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block5_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block5_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_3 (Add)                     (None, 19, 19, 728)  0           block5_sepconv3_bn[0][0]         
                                                                     add_2[0][0]                      
    __________________________________________________________________________________________________
    block6_sepconv1_act (Activation (None, 19, 19, 728)  0           add_3[0][0]                      
    __________________________________________________________________________________________________
    block6_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block6_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block6_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block6_sepconv2_act (Activation (None, 19, 19, 728)  0           block6_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block6_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block6_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block6_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block6_sepconv3_act (Activation (None, 19, 19, 728)  0           block6_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block6_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block6_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block6_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block6_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_4 (Add)                     (None, 19, 19, 728)  0           block6_sepconv3_bn[0][0]         
                                                                     add_3[0][0]                      
    __________________________________________________________________________________________________
    block7_sepconv1_act (Activation (None, 19, 19, 728)  0           add_4[0][0]                      
    __________________________________________________________________________________________________
    block7_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block7_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block7_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block7_sepconv2_act (Activation (None, 19, 19, 728)  0           block7_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block7_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block7_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block7_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block7_sepconv3_act (Activation (None, 19, 19, 728)  0           block7_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block7_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block7_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block7_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block7_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_5 (Add)                     (None, 19, 19, 728)  0           block7_sepconv3_bn[0][0]         
                                                                     add_4[0][0]                      
    __________________________________________________________________________________________________
    block8_sepconv1_act (Activation (None, 19, 19, 728)  0           add_5[0][0]                      
    __________________________________________________________________________________________________
    block8_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block8_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block8_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block8_sepconv2_act (Activation (None, 19, 19, 728)  0           block8_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block8_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block8_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block8_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block8_sepconv3_act (Activation (None, 19, 19, 728)  0           block8_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block8_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block8_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block8_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block8_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_6 (Add)                     (None, 19, 19, 728)  0           block8_sepconv3_bn[0][0]         
                                                                     add_5[0][0]                      
    __________________________________________________________________________________________________
    block9_sepconv1_act (Activation (None, 19, 19, 728)  0           add_6[0][0]                      
    __________________________________________________________________________________________________
    block9_sepconv1 (SeparableConv2 (None, 19, 19, 728)  536536      block9_sepconv1_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv1_bn (BatchNormal (None, 19, 19, 728)  2912        block9_sepconv1[0][0]            
    __________________________________________________________________________________________________
    block9_sepconv2_act (Activation (None, 19, 19, 728)  0           block9_sepconv1_bn[0][0]         
    __________________________________________________________________________________________________
    block9_sepconv2 (SeparableConv2 (None, 19, 19, 728)  536536      block9_sepconv2_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv2_bn (BatchNormal (None, 19, 19, 728)  2912        block9_sepconv2[0][0]            
    __________________________________________________________________________________________________
    block9_sepconv3_act (Activation (None, 19, 19, 728)  0           block9_sepconv2_bn[0][0]         
    __________________________________________________________________________________________________
    block9_sepconv3 (SeparableConv2 (None, 19, 19, 728)  536536      block9_sepconv3_act[0][0]        
    __________________________________________________________________________________________________
    block9_sepconv3_bn (BatchNormal (None, 19, 19, 728)  2912        block9_sepconv3[0][0]            
    __________________________________________________________________________________________________
    add_7 (Add)                     (None, 19, 19, 728)  0           block9_sepconv3_bn[0][0]         
                                                                     add_6[0][0]                      
    __________________________________________________________________________________________________
    block10_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_7[0][0]                      
    __________________________________________________________________________________________________
    block10_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block10_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block10_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block10_sepconv2_act (Activatio (None, 19, 19, 728)  0           block10_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block10_sepconv2 (SeparableConv (None, 19, 19, 728)  536536      block10_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv2_bn (BatchNorma (None, 19, 19, 728)  2912        block10_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block10_sepconv3_act (Activatio (None, 19, 19, 728)  0           block10_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block10_sepconv3 (SeparableConv (None, 19, 19, 728)  536536      block10_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block10_sepconv3_bn (BatchNorma (None, 19, 19, 728)  2912        block10_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_8 (Add)                     (None, 19, 19, 728)  0           block10_sepconv3_bn[0][0]        
                                                                     add_7[0][0]                      
    __________________________________________________________________________________________________
    block11_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_8[0][0]                      
    __________________________________________________________________________________________________
    block11_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block11_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block11_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block11_sepconv2_act (Activatio (None, 19, 19, 728)  0           block11_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block11_sepconv2 (SeparableConv (None, 19, 19, 728)  536536      block11_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv2_bn (BatchNorma (None, 19, 19, 728)  2912        block11_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block11_sepconv3_act (Activatio (None, 19, 19, 728)  0           block11_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block11_sepconv3 (SeparableConv (None, 19, 19, 728)  536536      block11_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block11_sepconv3_bn (BatchNorma (None, 19, 19, 728)  2912        block11_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_9 (Add)                     (None, 19, 19, 728)  0           block11_sepconv3_bn[0][0]        
                                                                     add_8[0][0]                      
    __________________________________________________________________________________________________
    block12_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_9[0][0]                      
    __________________________________________________________________________________________________
    block12_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block12_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block12_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block12_sepconv2_act (Activatio (None, 19, 19, 728)  0           block12_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block12_sepconv2 (SeparableConv (None, 19, 19, 728)  536536      block12_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv2_bn (BatchNorma (None, 19, 19, 728)  2912        block12_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block12_sepconv3_act (Activatio (None, 19, 19, 728)  0           block12_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    block12_sepconv3 (SeparableConv (None, 19, 19, 728)  536536      block12_sepconv3_act[0][0]       
    __________________________________________________________________________________________________
    block12_sepconv3_bn (BatchNorma (None, 19, 19, 728)  2912        block12_sepconv3[0][0]           
    __________________________________________________________________________________________________
    add_10 (Add)                    (None, 19, 19, 728)  0           block12_sepconv3_bn[0][0]        
                                                                     add_9[0][0]                      
    __________________________________________________________________________________________________
    block13_sepconv1_act (Activatio (None, 19, 19, 728)  0           add_10[0][0]                     
    __________________________________________________________________________________________________
    block13_sepconv1 (SeparableConv (None, 19, 19, 728)  536536      block13_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block13_sepconv1_bn (BatchNorma (None, 19, 19, 728)  2912        block13_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block13_sepconv2_act (Activatio (None, 19, 19, 728)  0           block13_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block13_sepconv2 (SeparableConv (None, 19, 19, 1024) 752024      block13_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    block13_sepconv2_bn (BatchNorma (None, 19, 19, 1024) 4096        block13_sepconv2[0][0]           
    __________________________________________________________________________________________________
    conv2d_3 (Conv2D)               (None, 10, 10, 1024) 745472      add_10[0][0]                     
    __________________________________________________________________________________________________
    block13_pool (MaxPooling2D)     (None, 10, 10, 1024) 0           block13_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    batch_normalization_3 (BatchNor (None, 10, 10, 1024) 4096        conv2d_3[0][0]                   
    __________________________________________________________________________________________________
    add_11 (Add)                    (None, 10, 10, 1024) 0           block13_pool[0][0]               
                                                                     batch_normalization_3[0][0]      
    __________________________________________________________________________________________________
    block14_sepconv1 (SeparableConv (None, 10, 10, 1536) 1582080     add_11[0][0]                     
    __________________________________________________________________________________________________
    block14_sepconv1_bn (BatchNorma (None, 10, 10, 1536) 6144        block14_sepconv1[0][0]           
    __________________________________________________________________________________________________
    block14_sepconv1_act (Activatio (None, 10, 10, 1536) 0           block14_sepconv1_bn[0][0]        
    __________________________________________________________________________________________________
    block14_sepconv2 (SeparableConv (None, 10, 10, 2048) 3159552     block14_sepconv1_act[0][0]       
    __________________________________________________________________________________________________
    block14_sepconv2_bn (BatchNorma (None, 10, 10, 2048) 8192        block14_sepconv2[0][0]           
    __________________________________________________________________________________________________
    block14_sepconv2_act (Activatio (None, 10, 10, 2048) 0           block14_sepconv2_bn[0][0]        
    __________________________________________________________________________________________________
    avg_pool (GlobalAveragePooling2 (None, 2048)         0           block14_sepconv2_act[0][0]       
    __________________________________________________________________________________________________
    predictions (Dense)             (None, 1000)         2049000     avg_pool[0][0]                   
    ==================================================================================================
    Total params: 22,910,480
    Trainable params: 22,855,952
    Non-trainable params: 54,528
    __________________________________________________________________________________________________
    

Now we have to calculate the gradient of the class output with respect to the convolutional layer output.


```python
with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    pred_index = tf.argmax(preds[0])
    class_channel = preds[:, pred_index]
```

Get the gradient of the output neuron with respect to the convolutional layer output.


```python
grads = tape.gradient(class_channel, last_conv_layer_output)
```

Let's look at these gradients.


```python
plt.hist(grads.numpy().flatten());
```


    
![png](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_33_0.png)
    


Now let's average the gradient over all the channels in the convolutional layer output


```python
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
```

Now we can compute the weighted summ of the convolutional layer output with respect to the averaged gradient. Then we sum all the channels to obtain the heatmap class activation.


```python
last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
```

Then normalize the heatmap to values between 0 and 1 to make it easier to visualize.


```python
heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
heatmap = heatmap.numpy()
```

Now let's take a look at what we've got.


```python
plt.matshow(heatmap);
```


    
![png](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_41_0.png)
    


That's good. We've got a bit of work to do to display this though. We've got to resize, smooth, and overlay it so that we can really understand it. We'll use a function from the keras tutorial to do that.


```python
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = load_img(img_path)
    img = img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))
```


```python
save_and_display_gradcam(img_path, heatmap)
```


    
![jpeg](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_44_0.jpg)
    


We can also use this to look for specific classes within the image. Here are some relevant ImageNet class indexes that we can look for. You can get the full list [here](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/).


```python
GOOSE_INDEX = 99
VIZSLA_INDEX = 211
GERMAN_SHAPARD_INDEX = 235
GREAT_DANE_INDEX = 246
CHOW_INDEX = 260
TABBY_CAT_INDEX = 281
TIGER_CAT_INDEX = 282
EGYPTIAN_CAT_INDEX = 285
```

We can put the work we did above into a function.


```python
def make_gradcam_heatmap(img_array: np.array, model, last_conv_layer_name: str, pred_index: int) -> np.ndarray:
    """
    This function creates a heatmap using Grad-CAM.
    """
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```


```python
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=VIZSLA_INDEX)

save_and_display_gradcam(img_path, heatmap, cam_path='dog_focus.jpg')
```


    
![jpeg](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_49_0.jpg)
    



```python
heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=TABBY_CAT_INDEX)

save_and_display_gradcam(img_path, heatmap, cam_path='cat_focus.jpg')
```


    
![jpeg](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_50_0.jpg)
    


Now let's put it all together.


```python
# Load the images
image1 = mpimg.imread('cat_and_dog_hats.png')
image2 = mpimg.imread('dog_focus.jpg')
image3 = mpimg.imread('cat_focus.jpg')

# Display the images
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,4))
ax1.imshow(image1)
ax2.imshow(image2)
ax3.imshow(image3)

# Add labels
ax1.set_xlabel('Original Image', fontsize=14)
ax2.set_xlabel('What Made You Think Vizsla?', fontsize=14)
ax3.set_xlabel('What Made You Think Tabby Cat?', fontsize=14)

# Turn off the tick marks and labels
ax1.set_xticks([])
ax1.set_yticks([])
ax2.set_xticks([])
ax2.set_yticks([])
ax3.set_xticks([])
ax3.set_yticks([])

# Turn off the grid
ax1.grid(visible=False)
ax2.grid(visible=False)
ax3.grid(visible=False)

# Show the plot
plt.show()
```


    
![png](2023-01-01-model-explainability-with-grad-cam_files/2023-01-01-model-explainability-with-grad-cam_52_0.png)
    



```python

```
