---
layout: post
title: "FastAI Notes and Thoughts"
description: "This post describes some of my notes and thoughts about the FastAI library"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/cheetah.jpg"
tags: [FastAI, Python]
---

This post is a collection of some notes and thoughts I've had when working with [FastAI](https://www.fast.ai/).

FastAI Models.

## Working on Windows

There seems to be an issue when training some models on Windows machines that I haven't run into when I've used Mac or Linux. Let's create a simple example to start.


```python
from fastai.vision.all import *
```


```python
path = untar_data(URLs.PETS)
files = get_image_files(path/"images")
def label_func(f):
    return f[0].isupper()
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
learn = cnn_learner(dls, resnet34, metrics=error_rate)
```

    Due to IPython and Windows limitation, python multiprocessing isn't available now.
    So `number_workers` is changed to 0 to avoid getting stuck
    

When I try to train this model, I run into an `OSError`.


```python
try:
    learn.fine_tune(1)
except OSError as err:
    print(f"Error! You have the following error: {err}")
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>


    Error! You have the following error: image file is truncated (7 bytes not processed)
    

The solution is to add the following before training your model: 


```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```