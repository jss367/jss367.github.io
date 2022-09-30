---
layout: post
title: "Siamese Networks with FastAI"
description: "This tutorial describes how to work with the FastAI library for siamese networks"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/platypus2.jpg"
tags: [FastAI, Neural Networks, Python]
---

This post shows how to load and run inference with the model we built in the [previous post](https://jss367.github.io/siamese-network-tutorial-with-fastai-update.html).


```python
from fastai.vision.all import *
import dill
from siam_utils import SiameseImage
```


```python
learner = load_learner(Path(os.getenv("MODELS")) / "siam_catsvdogs_tutorial.pkl", cpu=False, pickle_module=dill)
```


```python
path = untar_data(URLs.PETS)
files = get_image_files(path / "images")
```


```python
imgval = PILImage.create(files[0])
imgtest = PILImage.create(files[1])
siamtest = SiameseImage(imgval, imgtest)
```


```python
siamtest.show();
```


    
![png](2022-09-30-siamese-network-tutorial-with-fastai-loading-model_files/2022-09-30-siamese-network-tutorial-with-fastai-loading-model_6_0.png)
    



```python
@patch
def siampredict(self: Learner, item, rm_type_tfms=None, with_input=False):
    res = self.predict(item, rm_type_tfms=None, with_input=False)
    if res[0] == tensor(0):
        SiameseImage(item[0], item[1], "Prediction: Not similar").show()
    else:
        SiameseImage(item[0], item[1], "Prediction: Similar").show()
    return res
```


```python
res = learner.siampredict(siamtest)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](2022-09-30-siamese-network-tutorial-with-fastai-loading-model_files/2022-09-30-siamese-network-tutorial-with-fastai-loading-model_8_2.png)
    



```python
imgval = PILImage.create(files[9])
imgtest = PILImage.create(files[35])
siamtest = SiameseImage(imgval, imgtest)
```


```python
siamtest.show();
```


    
![png](2022-09-30-siamese-network-tutorial-with-fastai-loading-model_files/2022-09-30-siamese-network-tutorial-with-fastai-loading-model_10_0.png)
    



```python
res = learner.siampredict(siamtest)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








    
![png](2022-09-30-siamese-network-tutorial-with-fastai-loading-model_files/2022-09-30-siamese-network-tutorial-with-fastai-loading-model_11_2.png)
    


Oh well, it's not perfect.


```python

```
