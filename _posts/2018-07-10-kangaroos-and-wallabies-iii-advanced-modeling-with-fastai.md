---
layout: post
title: "Kangaroos and Wallabies III: Advanced Modeling with FastAI"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/wallaby_cliff.jpg"
tags: [Computer Vision, Convolutional Neural Networks, Deep Learning, FastAI, Neural Networks, Wildlife]
---

In this notebook, we're going to take the [dataset we prepared] and continue to iterate on the modeling. Last time [we built a model using TensorFlow and Xception](https://jss367.github.io/kangaroos-and-wallabies-ii-building-a-model.html). This time, we're going to iterate on that using [FastAI](https://github.com/fastai/fastai).

> Note: This notebook has been rewritten since FastAI has changed so much. See below for the most recent version it was run with.


```python
import fastai
from fastai.data.all import *
from fastai.vision.all import *
```


```python
fastai.__version__
```




    '2.5.3'



First, we need to create a `DataBlock`. Please see my [FastAI Data Tutorial - Image Classification](https://jss367.github.io/fastai-data-tutorial-image-classification.html) for details on this process.


```python
dblock = DataBlock(blocks    = (ImageBlock, CategoryBlock),
                   get_items = get_image_files,
                   get_y     = parent_label,
                   splitter  = GrandparentSplitter('train', 'val'),
                   item_tfms = Resize(224))
```

We need to point to the image root path.


```python
if sys.platform == 'linux':
    path = Path(r'/home/julius/data/WallabiesAndRoosFullSize')
else:
    path = Path(r'E:/Data/Raw/WallabiesAndRoosFullSize')
```


```python
dls = dblock.dataloaders(path)
```

Let’s see how many items we have in each set.


```python
len(dls.train.items), len(dls.valid.items)
```




    (3653, 567)



Let's look at some example images to make sure everything is correct.


```python
dls.train.show_batch(max_n=4, nrows=1)
```


    
![png]({{site.baseurl}}/2018-07-10-kangaroos-and-wallabies-iii-advanced-modeling-with-fastai_files/2018-07-10-kangaroos-and-wallabies-iii-advanced-modeling-with-fastai_13_0.png)
    


FastAI provides some standard transforms that work quite well, so we'll use them.


```python
dblock = dblock.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'), batch_tfms=aug_transforms(mult=2))
dls = dblock.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)
```

    /home/julius/miniconda3/envs/fai/lib/python3.8/site-packages/torch/_tensor.py:1023: UserWarning: torch.solve is deprecated in favor of torch.linalg.solveand will be removed in a future PyTorch release.
    torch.linalg.solve has its arguments reversed and does not return the LU factorization.
    To get the LU factorization see torch.lu, which can be used with torch.lu_solve or torch.lu_unpack.
    X = torch.solve(B, A).solution
    should be replaced with
    X = torch.linalg.solve(A, B) (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/aten/src/ATen/native/BatchLinearAlgebra.cpp:760.)
      ret = func(*args, **kwargs)
    


    
![png]({{site.baseurl}}/2018-07-10-kangaroos-and-wallabies-iii-advanced-modeling-with-fastai_files/2018-07-10-kangaroos-and-wallabies-iii-advanced-modeling-with-fastai_15_1.png)
    



```python
learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)
```

    /home/julius/miniconda3/envs/fai/lib/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448234945/work/c10/core/TensorImpl.h:1156.)
      return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
    


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
    <tr>
      <td>0</td>
      <td>0.720236</td>
      <td>0.259093</td>
      <td>0.104056</td>
      <td>02:09</td>
    </tr>
  </tbody>
</table>



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
    <tr>
      <td>0</td>
      <td>0.339751</td>
      <td>0.301955</td>
      <td>0.141093</td>
      <td>02:07</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.225766</td>
      <td>0.444917</td>
      <td>0.105820</td>
      <td>02:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.146328</td>
      <td>0.176450</td>
      <td>0.095238</td>
      <td>02:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.109386</td>
      <td>0.162337</td>
      <td>0.091711</td>
      <td>02:06</td>
    </tr>
  </tbody>
</table>


The `error_rate` pertains to the validation set, showing that we got about 91%. Let's look at a confusion matrix.


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
```






    
![png]({{site.baseurl}}/2018-07-10-kangaroos-and-wallabies-iii-advanced-modeling-with-fastai_files/2018-07-10-kangaroos-and-wallabies-iii-advanced-modeling-with-fastai_18_1.png)
    

