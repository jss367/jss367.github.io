---
layout: post
title: "Exploring Images with Python II"
thumbnail: "assets/img/mining_landscape.jpg"
feature-img: "assets/img/rainbow.jpg"
tags: [Computer Vision, Python]
---

This post shows some of the various tools in Python for visualizing images. There are usually two steps to the visualization process. First, you'll need to read in the image, usually as a `numpy` array or something similar. Then, you can visualize it with various libraries.

<b>Table of contents</b>
* TOC
{:toc}

## SKImage

SKImage is used for turning an image on disk into a numpy array, like so.


```python
import skimage
from skimage.io import imread
```


```python
image_path = '../roo.jpg'
```


```python
img = imread(image_path)
```


```python
img[:, : , 0][:3]
```




    array([[232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
            232, 232, 232, 234, 234, 234, 234, 234, 234, 234, 234, 231, 231,
            231, 231, 231, 231, 231, 231, 234, 234, 234, 234, 234, 234, 234,
            234, 237, 237, 237, 237, 237, 237, 237, 237, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 234, 234, 234, 234, 234, 234, 234, 234, 232,
            232, 232, 232, 232, 232, 232, 232, 229, 229, 229, 229, 229, 229,
            229, 229, 227, 227, 227, 227, 227, 227, 227, 227],
           [232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
            232, 232, 232, 234, 234, 234, 234, 234, 234, 234, 234, 232, 232,
            232, 232, 232, 232, 232, 232, 234, 234, 234, 234, 234, 234, 234,
            234, 237, 237, 237, 237, 237, 237, 237, 237, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 234, 234, 234, 234, 234, 234, 234, 234, 232,
            232, 232, 232, 232, 232, 232, 232, 229, 229, 229, 229, 229, 229,
            229, 229, 227, 227, 227, 227, 227, 227, 227, 227],
           [232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232, 232,
            232, 232, 232, 234, 234, 234, 234, 234, 234, 234, 234, 232, 232,
            232, 232, 232, 232, 232, 232, 234, 234, 234, 234, 234, 234, 234,
            234, 237, 237, 237, 237, 237, 237, 237, 237, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241, 241,
            241, 241, 241, 241, 234, 234, 234, 234, 234, 234, 234, 234, 232,
            232, 232, 232, 232, 232, 232, 232, 229, 229, 229, 229, 229, 229,
            229, 229, 227, 227, 227, 227, 227, 227, 227, 227]], dtype=uint8)



## Matplotlib


```python
from matplotlib import pyplot as plt
```

Matplotlib is my default for displaying images. Normally when you run a command in Jupyter Notebooks it will print the last output. But the output isn't very useful so to suppress it you can end the line with a semicolon.


```python
plt.imshow(img);
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_12_0.png)


To have more control:


```python
fix, ax = plt.subplots()
ax.imshow(img);
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_14_0.png)



```python
plt.figure(figsize=(10, 10))
plt.imshow(img);
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_15_0.png)



```python
fig, ax = plt.subplots(figsize=(30, 10))
ax.imshow(img)
ax.axis('off');
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_16_0.png)



```python
plt.rcParams['figure.figsize'] = (30, 10)
fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off');
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_17_0.png)


# PIL


```python
from PIL import Image
```


```python
pil_img = Image.fromarray(img)
```


```python
pil_img
```




![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_21_0.png)



# Open CV

You can also read images off disk using OpenCV.


```python
import cv2
```


```python
img = cv2.imread(image_path)
```


```python
plt.imshow(img);
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_26_0.png)


Note the flipped color channels. Here is how you can fix it.


```python
plt.imshow(img[:, :, ::-1]);
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_28_0.png)


## ImageIO

ImageIO is nice because it has a common interface for different image types.


```python
import imageio
```


```python
image_path = '../roo.jpg'
```


```python
img_arr = imageio.imread(image_path)
```


```python
img_arr.shape
```




    (256, 192, 3)



The three axes are the height, width, and number of color channels. So is image is 256 pixels high, 192 pixels wide, and has 3 color channels.


```python
type(img_arr)
```




    imageio.core.util.Array



This is like a NumPy array.

# Plotting Multiple Images

Here's an example of plotting multiple images with the label below it. It's commonly done to visualize datasets.


```python
import tensorflow as tf

from tensorflow.keras import datasets
import matplotlib.pyplot as plt
```


```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


train_images, test_images = train_images / 255.0, test_images / 255.0
```


```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()
```


![png](2018-06-06-Exploring%20Images%20with%20Python%20II_files/2018-06-06-Exploring%20Images%20with%20Python%20II_42_0.png)

