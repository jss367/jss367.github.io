---
layout: post
title: "PyTorch Notes"
description: "Some notes on using PyTorch after working with TensorFlow and Keras."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/fire.jpg"
tags: [Python, PyTorch]
---

This posts contains some of my notes from switching to PyTorch after having worked with TensorFlow and Keras for a long time.

<b>Table of Contents</b>
* TOC
{:toc}


```python
import imageio
import torch
```



## Channels First

PyTorch requires channels first, so you may have to use the `permute` method to get your images in the right shape.


```python
image_path = 'roo.jpg'
```


```python
img_arr = imageio.imread(image_path)
```


```python
img = torch.from_numpy(img_arr)
```


```python
img.shape
```




    torch.Size([256, 192, 3])




```python
fixed_img = img.permute(2, 0, 1)
```


```python
fixed_img.shape
```




    torch.Size([3, 256, 192])



Note that `fixed_img` isn't a copy of the original, it's just a reshaping. So if you go on to change `img`, `fixed_img` will change as well.

## Missing from PyTorch

#### model.summary()

There is no `model.summary()` like there is in Keras, so instead I recommend [torchinfo](https://github.com/TylerYep/torchinfo). This is a great tool for sanity checking a network.
