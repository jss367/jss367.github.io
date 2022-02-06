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


The solution is to add the following before training your model: 


```python
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```


```python
try:
    learn.fine_tune(1)
except OSError as err:
    print(f"Error! You have the following error: {err}")
```



Now it works.

## Models

One thing I really like is that the `learn.model` is a PyTorch object, which makes it immediately familiar to anyone working in PyTorch.


```python
type(learn.model)
```




    torch.nn.modules.container.Sequential



This means that everything you would normally do with a PyTorch model, you can do with a FastAI model.


```python
learn.model.eval();
```

## Transitioning From FastAI Version 1

For those of you who used the first version of FastAI, there are a lot of differences. When loading data, you might be looking for `DataBunch`es. Those no longer exist, but you will see similar functionality in the `DataBlock`s. Also, lots of smaller data classes, such as `ImageList` and `ImageImageList`, no longer exist. Check out my [data tutorial](https://jss367.github.io/fastai-data-tutorial-image-classification.html) to see how to work with `DataBlock`s.

Also, the `import *` statement has changed.
```python
from fastai.vision import *
```
is now
```python
from fastai.vision.all import *
```

## There's Hidden Stuff All Around

There's a lot of cool features hidden around FastAI. These are great when you know they're there, but surprising when you run into one unexpectedly.

Let's say you're got a model in at `analysis/catsvdogs/models/my_model` that you want to load. So you create a `Path` with it and point `learn.load` to it. But instead of loading, this causes an error.


```python
path = Path('analysis/catsvdogs/models/my_model')
try:
    learn.load(path)
except FileNotFoundError as err:
    print(err)
```

    [Errno 2] No such file or directory: 'C:\\Users\\Julius\\.fastai\\data\\oxford-iiit-pet\\models\\analysis\\catsvdogs\\models\\my_model.pth'
    

It adds the word "model" in the beginning. This is great if you know it's going to do that, but not for people who aren't expecting that. It's not even obvious when you look at the default arguments how to turn this off. This is a minor annoyance, but there are a lot of these. I have found that FastAI seems to make more assumptions about what you want relative to other libraries.

## Things I don't like

There are a lot of things I like about FastAI, but there are also some things that I don't.

#### Evaluation

I find the way evaluation works in FastAI to be counterintuitive. If you're familiar with `keras`, you're used to calling `.evaluate` on a model after it's been trained. But in `fastai`, there are only two splits - train and val. So if you want to evaluate on test data, you have to create a new `DataBunch` with training data, then swap in your test data by calling it validation data. At least, that's the way I've been getting it to work. If I find a better way, I'll update this post.

#### Variable Names

The first one that sticks out is the use of single-letter variables. For me, this adds confusion and does not save time. There are lots of these. For example, take a look at the [`params_size` function](https://github.com/fastai/fastai/blob/54a9e3cf4fd0fa11fc2453a5389cc9263f6f0d77/fastai/callbacks/hooks.py#L136), which takes an input `m`. It's not obvious to me what `m` refers to. I find that many times I have to poke around a bit to see what functions are expecting, mostly because I can't figure it out from the argument names.

It's almost like it's designed for people who are going to use the library all the time and know it intimately, but difficult for anyone else.

#### Splitters

Datasets require splitters. I don't mind this, but I think there should be a way to not include a splitter. It's not that I don't think splitting between training and validation sets is a good idea, it's just that there are some cases when, for whatever reason, that's not what I'm trying to do at the moment. But it seems like when you try to go your own way, it's not easy.

#### Namespace Collisions

One thing that I know annoys people about FastAI is the liberal use of `import *` (as I did at the top of the notebook). It's kind of a Python anti-pattern so to see it everywhere can be disconcerting.

I see where the annoyance comes from. You have no idea what's in your namespace, and that's a problem. For example, `image_size` is a function in FastAI that is automatically imported. I often do something like `image_size=(244, 244)` in my code, but if I do this I'm overwriting a function that could be used. This leads to nasty bugs.

#### Inheriting from Outer Context

There's a lot of inheriting from outer context in FastAI. This works well for a Jupyter Notebook environment, but makes it harder to use it in production.

## Conclusion

I didn't want to end on a negative so I want to say that there's a lot I really like about FastAI. I hope the library continues to be developed. I've noticed that there's not a big development environment around it - right now there's only one core developer on it, and keeping a library going is a lot of work for a single person.
