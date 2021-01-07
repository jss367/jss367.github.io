---
layout: post
title: "Data Augmentation with Albumentations"
description: "A walkthrough with lots of images of the albumentations library for data augmentation"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/dark_path.jpg"
tags: [Python, Data Augmentation, Deep Learning]
---

This post is going to demonstrate the [albumentations](https://albumentations.ai/) library.


```python
import random
from typing import List

import albumentations as A
import cv2
import imageio
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
```

To illustrate the data augmentation techniques, I'm going to use an image from the [semantic drone dataset from Kaggle](https://www.kaggle.com/bulentsiyah/semantic-drone-dataset#). Here's the [link to the original version](https://www.tugraz.at/index.php?id=22387).

Let's read in the image.


```python
image = imageio.imread('semseg_image.jpg')
mask = imageio.imread('semseg_label.png')
```

Now we have two numpy arrays, which are the best format to feed into Albumentations.


```python
print(isinstance(image, np.ndarray))
print(isinstance(mask, np.ndarray))
print(image.shape, mask.shape)
```

    True
    True
    (4000, 6000, 3) (4000, 6000)
    

The mask has different values for the different classes.


```python
np.unique(mask, return_counts=True)
```




    (Array([ 0,  1,  2,  4,  8, 10, 15, 19, 22], dtype=uint8),
     array([   60393, 20883680,   293514,   633602,   211227,   287613,
              524123,   589714,   516134], dtype=int64))



OpenCV uses BGR internally. This usually doesn't come up unless you read an image with opencv and try to plot it with, say matplotlib. Then you'll need to switch the channels around.


```python
original_height, original_width = image.shape[:2]
print(original_height)
print(original_width)
```

    4000
    6000
    

The image is bigger than we'll need, so let's resize it first.


```python
downsize_factor = 6
image = image[::downsize_factor, ::downsize_factor]
mask = mask[::downsize_factor, ::downsize_factor]
```

Let's take a look at the image and the mask.


```python
def visualize(image, mask, original_image=None, original_mask=None, fontsize = 18):

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image)
        ax[1].imshow(mask)

    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)
        
        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)
        
        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)
        
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
```

Now let's look at a bunch of augmentations.

### Spatial augmentations

There are eight different alignments that an image could be in without losing any of the image and my goal is to make each of them equally likely. Using the following setup, each position will have a 12.5% chance of being selected.


```python
augments = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    
])

for i in range(10):
    augmented = augments(image=image, mask=mask)

    image_scaled = augmented['image']
    mask_scaled = augmented['mask']

    visualize(image_scaled, mask_scaled, original_image=image, original_mask=mask)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_0.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_1.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_2.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_3.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_4.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_5.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_6.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_7.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_8.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_18_9.png)
    


There is also `A.RandomRotate90(p=0.5)` but I don't recommend it. The above are all you need

## Pixel-level augmentations

These by default don't change the segmentation label, so we won't need to include that.

### CLAHE

Contrast Limited Adaptive Histogram Equalization (CLAHE) is a good option. I usually include it and leave it in the defaults. This by default does a range, so you'll have to see it multiple times

To specify this, you usually set a clip limit and it randomly chooses a value between 1 and your specified limit. This means that you get a different value each time you run it. To fix it to be the same, I'm going to set the value as both the high and the low to see that exact value. In practice I wouldn't do this.

This type of augmentation doesn't change the mask, so we can ignore it for now.


```python
def visualize_no_mask(augmented_images: List, fontsize = 18):
    
    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(3, 3),
                     axes_pad=0.5,
                     )

    for ax, im in zip(grid, augmented_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        #ax.set_title('Augmented Image', fontsize=fontsize)

    plt.show()
```


```python
augments = A.Compose([
        A.CLAHE(clip_limit=50, p=1)
])
aug_images = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_images.append(augmented['image'])

visualize_no_mask(aug_images)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_27_0.png)
    


And here it is without specifying the `clip_limit`.


```python
augments = A.Compose([
        A.CLAHE(p=0.8),
])
aug_images = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_images.append(augmented['image'])

visualize_no_mask(aug_images)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_29_0.png)
    


### Equalize

Another good one is equalize. This just equalizes the image histogram. I wouldn't use it during training unless you also planned to use it during test time. Again it's not going to change the mask.


```python
augments = A.Compose([
        A.Equalize(p=1),
])

aug_ims = []
augmented = augments(image=image, mask=mask)

aug_ims.append(augmented['image'])
augmented_image = augmented['image']
augmented_mask = augmented['mask']

visualize(image, mask, original_image=image, original_mask=mask)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_32_0.png)
    



```python
augments = A.Compose([
        A.Equalize(p=1),
])

aug_images = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_images.append(augmented['image'])

visualize_no_mask(aug_images)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_33_0.png)
    


## Posterize

This isn't that obvious to see but it might be a good idea. I don't use it much. It was used in the [AutoAugment research by Google Brain](https://arxiv.org/pdf/1805.09501.pdf).


```python
augments = A.Compose([
        A.Posterize(num_bits=4, p=0.5),
])

augmented = augments(image=image, mask=mask)

image_scaled = augmented['image']
mask_scaled = augmented['mask']

visualize(image_scaled, mask_scaled, original_image=image, original_mask=mask)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_36_0.png)
    


### RandomBrightness


```python
augments = A.Compose([
        A.RandomBrightness(limit=(0.3, 0.3), p=1),
])

aug_ims = []
augmented = augments(image=image, mask=mask)

aug_ims.append(augmented['image'])
augmented_image = augmented['image']
augmented_mask = augmented['mask']

visualize(image, mask, original_image=image, original_mask=mask)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_38_0.png)
    


If you set a single value it will be plus or minus this value, which is what you'd actually want. Because it's a range I leave the probability at 100%.


```python
augments = A.Compose([
        A.RandomBrightness(limit=0.3, p=1),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_40_0.png)
    


### RandomConstrast

`RandomContrast` is similar to `RandomBrightness` but I find you can get away with doing a bit more.


```python
augments = A.Compose([
        A.RandomContrast(limit=0.5, p=1),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_43_0.png)
    


### RandomGamma


```python
augments = A.Compose([
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_45_0.png)
    



```python

```


```python

```

There are also a lot of distortions. Most of them seem like to significant a change for me. If you want a smaller one, you can always try `GridDistortion`.


```python

```

Another thing to do during augmentation is convert everything to float... Is this needed?


```python
A.ToFloat(max_value=255.0, p=1.0)
```




    ToFloat(always_apply=False, p=1.0, max_value=255.0)




```python

```

## Cropping

There are a few cropping options: `Crop`, `CenterCrop`, and `RandomResizedCrop`. Let's look at them.


```python
aug = A.Compose([
        A.Crop(40, 40, 120, 120, p=0.5),
])
```

This reduces the image size. Sometimes this is what I'm looking for, other times I want to resize.


```python
plt.imshow(aug(image=image, mask=mask)['image'])
```




    <matplotlib.image.AxesImage at 0x274a14e9bb0>




    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_57_1.png)
    



```python
plt.imshow(aug(image=image, mask=mask)['image'])
```




    <matplotlib.image.AxesImage at 0x274a60eb5b0>




    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_58_1.png)
    


Here's how to resize.

The arguments are a little unintuitive to me but they work well.


```python
w2h_ratio = image.shape[1] / image.shape[0]
print(w2h_ratio)
```

    1.4992503748125936
    


```python
augments = A.Compose([
        A.RandomResizedCrop(image.shape[0], image.shape[1]),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_62_0.png)
    


Compare the above to `Crop`, where the images aren't resized.


```python
augments = A.Compose([
        A.Crop(300, 300, 500, 500, p=0.5),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_64_0.png)
    


And here's `CenterCrop`.


```python
augments = A.Compose([
        A.CenterCrop(400, 400,always_apply=False, p=0.5),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_66_0.png)
    


## Overall

My overall usually looks something like this


```python
augments = A.Compose([
    # spatial-level transforms (no distortion)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    # pixel-level transforms
        A.CLAHE(p=0.5),
        A.Equalize(p=0.5),
        A.Posterize(num_bits=4, p=0.5),
        A.RandomBrightness(limit=0.2, p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_69_0.png)
    



```python
for i in range(10):
    augmented = augments(image=image, mask=mask)

    image_scaled = augmented['image']
    mask_scaled = augmented['mask']

    visualize(image_scaled, mask_scaled, original_image=image, original_mask=mask)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_0.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_1.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_2.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_3.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_4.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_5.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_6.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_7.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_8.png)
    



    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_70_9.png)
    


Or if you want to add cropping:


```python
augments = A.Compose([
    # spatial-level transforms (no distortion)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    # pixel-level transforms
        A.CLAHE(p=0.5),
        A.Equalize(p=0.5),
        A.Posterize(num_bits=4, p=0.5),
        A.RandomBrightness(limit=0.2, p=0.5),
        A.RandomContrast(limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    # cropping
        A.RandomResizedCrop(image.shape[0], image.shape[1])
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_72_0.png)
    



```python
augments = A.Compose([
        A.RandomResizedCrop(image.shape[0], image.shape[1]),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_73_0.png)
    


The default scale values can be too aggressive for me, so if you want to constrain it you can set the scale.


```python
augments = A.Compose([
        A.RandomResizedCrop(image.shape[0], image.shape[1], scale=(0.5, 1.0)),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_75_0.png)
    


If you don't want to change the aspect ratio, you can fix the ratio to 1.


```python
augments = A.Compose([
        A.RandomResizedCrop(image.shape[0], image.shape[1], scale=(0.5, 1.0), ratio=(1,1)),
])

aug_ims = []
for i in range(9):
    augmented = augments(image=image, mask=mask)

    aug_ims.append(augmented['image'])

visualize_no_mask(aug_ims)
```


    
![png](2020-06-12-Data-Augmentation-with-Albumentations_files/2020-06-12-Data-Augmentation-with-Albumentations_77_0.png)
    



```python

```
