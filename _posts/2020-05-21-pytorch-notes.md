---
layout: post
title: "PyTorch Notes"
description: "Some notes on using PyTorch after working with TensorFlow and Keras."
feature-img: "assets/img/rainbow.jpg"
tags: [Python, PyTorch]
---

This posts contains some of my notes from switching to PyTorch after having worked with TensorFlow and Keras for a long time.

<b>Table of Contents</b>
* TOC
{:toc}


```python
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
    

## Missing from PyTorch

#### model.summary()

There is no `model.summary()` like there is in Keras, so instead I recommend [torchinfo](https://github.com/TylerYep/torchinfo). This is a great tool for sanity checking a network.
