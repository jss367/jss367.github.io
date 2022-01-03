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

## Accessing the GPU


```python
torch.cuda.current_device()
```




    0



How many are available?


```python
torch.cuda.device_count()
```




    1



What's the name of the GPU I'm using?


```python
torch.cuda.get_device_name(0)
```




    'NVIDIA GeForce GTX 960'



Is a GPU available?


```python
torch.cuda.is_available()
```




    True



How much memory is being used?


```python
print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
```

    Allocated: 0.0 GB
    
## Specify which GPU to use

Just like in TensorFlow, you can specify which GPU to use with the following. Be sure to do this before you import TensorFlow/PyTorch.

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
