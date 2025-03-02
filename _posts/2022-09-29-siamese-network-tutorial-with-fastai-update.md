---
layout: post
title: "Siamese Networks with FastAI - Update"
description: "This tutorial describes how to work with the FastAI library for siamese networks"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/platypus1.jpg"
tags: [FastAI, Neural Networks, Python]
---

This is post is a walkthrough of creating a Siamese network with FastAI. I had planned to simply use the [tutorial from FastAI](https://docs.fast.ai/tutorial.siamese.html), but I had to change so much to be able to load the model and make it all work with the latest versions that I figured I would turn it into a blog post. This is really similar to my other post on [Siamese Networks with FastAI](https://jss367.github.io/siamese-networks-with-fastai.html), except that in this one I will follow on with a post about [how to evaluate the model](https://jss367.github.io/siamese-network-tutorial-with-fastai-evaluation.html).

<b>Table of Contents</b>
* TOC
{:toc}

The first major thing I had to do was create a file called `siam_utils.py` where I put all the code that I would need for both my training and inference. If I just copied the code to my inference notebook it wouldn't work, so this step is essential.

## siam_utils.py

Here is the code for `siam_utils.py`:

```python

import PIL
import re
from fastai.vision.all import *


class SiameseImage(fastuple):
    def show(self, ctx=None, **kwargs):
        if len(self) > 2:
            img1, img2, similarity = self
        else:
            img1, img2 = self
            similarity = "Undetermined"
        if not isinstance(img1, Tensor):
            if img2.size != img1.size:
                img2 = img2.resize(img1.size)
            t1, t2 = tensor(img1), tensor(img2)
            t1, t2 = t1.permute(2, 0, 1), t2.permute(2, 0, 1)
        else:
            t1, t2 = img1, img2
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1, line, t2], dim=2), title=similarity, ctx=ctx, **kwargs)


def open_image(fname, size=224):
    img = PIL.Image.open(fname).convert("RGB")
    img = img.resize((size, size))
    t = torch.Tensor(np.array(img))
    return t.permute(2, 0, 1).float() / 255.0


def label_func(fname):
    return re.match(r"^(.*)_\d+.jpg$", fname.name).groups()[0]


path = untar_data(URLs.PETS)
files = get_image_files(path / "images")
labels = list(set(files.map(label_func)))
lbl2files = {l: [f for f in files if label_func(f) == l] for l in labels}


class SiameseTransform(Transform):
    def __init__(self, files, splits):
        self.splbl2files = [{l: [f for f in files[splits[i]] if label_func(f) == l] for l in labels} for i in range(2)]
        self.valid = {f: self._draw(f, 1) for f in files[splits[1]]}

    def encodes(self, f):
        f2, same = self.valid.get(f, self._draw(f, 0))
        img1, img2 = PILImage.create(f), PILImage.create(f2)
        return SiameseImage(img1, img2, int(same))

    def _draw(self, f, split=0):
        same = random.random() < 0.5
        cls = label_func(f)
        if not same:
            cls = random.choice(L(l for l in labels if l != cls))
        return random.choice(self.splbl2files[split][cls]), same

```

Now on to this part. I haven't included much verbiage in this post. If you want to see the motivation behind some steps, I recommend you check out the [FastAI tutorial](https://docs.fast.ai/tutorial.siamese.html), which has the same steps in general.


```python
from fastai.vision.all import *
from siam_utils import *
```


```python
img = PIL.Image.open(files[0])
img
```




    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_8_0.png)
    




```python
open_image(files[0]).shape
```




    torch.Size([3, 224, 224])



## Writing your own data block


```python
class ImageTuple(fastuple):
    @classmethod
    def create(cls, fns):
        return cls(tuple(PILImage.create(f) for f in fns))

    def show(self, ctx=None, **kwargs):
        t1, t2 = self
        if not isinstance(t1, Tensor) or not isinstance(t2, Tensor) or t1.shape != t2.shape:
            return ctx
        line = t1.new_zeros(t1.shape[0], t1.shape[1], 10)
        return show_image(torch.cat([t1, line, t2], dim=2), ctx=ctx, **kwargs)
```


```python
img = ImageTuple.create((files[0], files[1]))
tst = ToTensor()(img)
type(tst[0]), type(tst[1])
```




    (fastai.torch_core.TensorImage, fastai.torch_core.TensorImage)




```python
img1 = Resize(224)(img)
tst = ToTensor()(img1)
tst.show();
```


    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_13_0.png)
    



```python
def ImageTupleBlock():
    return TransformBlock(type_tfms=ImageTuple.create, batch_tfms=IntToFloatTensor)
```


```python
splits = RandomSplitter()(files)
```


```python
splits_files = [files[splits[i]] for i in range(2)]
splits_sets = mapped(set, splits_files)
```


```python
def get_split(f):
    for i, s in enumerate(splits_sets):
        if f in s:
            return i
    raise ValueError(f"File {f} is not presented in any split.")
```


```python
splbl2files = [{l: [f for f in s if label_func(f) == l] for l in labels} for s in splits_sets]
```


```python
def splitter(items):
    def get_split_files(i):
        return [j for j, (f1, f2, same) in enumerate(items) if get_split(f1) == i]

    return get_split_files(0), get_split_files(1)
```


```python
def draw_other(f):
    same = random.random() < 0.5
    cls = label_func(f)
    split = get_split(f)
    if not same:
        cls = random.choice(L(l for l in labels if l != cls))
    return random.choice(splbl2files[split][cls]), same
```


```python
def get_tuples(files):
    return [[f, *draw_other(f)] for f in files]
```


```python
def get_x(t):
    return t[:2]


def get_y(t):
    return t[2]
```


```python
siamese = DataBlock(
    blocks=(ImageTupleBlock, CategoryBlock),
    get_items=get_tuples,
    get_x=get_x,
    get_y=get_y,
    splitter=splitter,
    item_tfms=Resize(224),
    batch_tfms=[Normalize.from_stats(*imagenet_stats)],
)
```


```python
dls = siamese.dataloaders(files)
```


```python
b = dls.one_batch()
explode_types(b)
```




    {tuple: [{__main__.ImageTuple: [fastai.torch_core.TensorImage,
        fastai.torch_core.TensorImage]},
      fastai.torch_core.TensorCategory]}




```python
@typedispatch
def show_batch(x: ImageTuple, y, samples, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None:
        figsize = (ncols * 6, max_n // ncols * 3)
    if ctxs is None:
        ctxs = get_grid(min(len(samples), max_n), nrows=nrows, ncols=ncols, figsize=figsize)
    ctxs = show_batch[object](x, y, samples, ctxs=ctxs, max_n=max_n, **kwargs)
    return ctxs
```


```python
dls.show_batch()
```


    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_27_0.png)
    



```python
class SiameseModel(Module):
    def __init__(self, encoder, head):
        self.encoder, self.head = encoder, head

    def forward(self, x1, x2):
        ftrs = torch.cat([self.encoder(x1), self.encoder(x2)], dim=1)
        return self.head(ftrs)
```


```python
# encoder = create_body(resnet34, cut=-2) # worked in old version of fastai/torchvision
encoder = create_body(resnet34(weights=ResNet34_Weights.IMAGENET1K_V1), cut=-2) # update for new torchvision
```


```python
head = create_head(512 * 2, 2, ps=0.5)
model = SiameseModel(encoder, head)
```


```python
def siamese_splitter(model):
    return [params(model.encoder), params(model.head)]
```


```python
def loss_func(out, targ):
    return CrossEntropyLossFlat()(out, targ.long())
```


```python
splits = RandomSplitter()(files)
tfm = SiameseTransform(files, splits)
tls = TfmdLists(files, tfm, splits=splits)
dls = tls.dataloaders(
    after_item=[Resize(224), ToTensor], after_batch=[IntToFloatTensor, Normalize.from_stats(*imagenet_stats)]
)
```


```python
valids = [v[0] for k, v in tfm.valid.items()]
assert not [v for v in valids if v in files[splits[0]]]
```

## Create a Learner


```python
learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), splitter=siamese_splitter, metrics=accuracy)
```


```python
learn.freeze()
```

## Train the Model


```python
learn.lr_find()
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










    SuggestedLRs(valley=0.0030199517495930195)




    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_39_3.png)
    



```python
learn.fit_one_cycle(4, 3e-3)
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




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.543962</td>
      <td>0.344742</td>
      <td>0.841678</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.367816</td>
      <td>0.221206</td>
      <td>0.919486</td>
      <td>00:21</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.298733</td>
      <td>0.179088</td>
      <td>0.935724</td>
      <td>00:22</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.252596</td>
      <td>0.172828</td>
      <td>0.937754</td>
      <td>00:22</td>
    </tr>
  </tbody>
</table>


## See the Results


```python
@typedispatch
def show_results(x: SiameseImage, y, samples, outs, ctxs=None, max_n=6, nrows=None, ncols=2, figsize=None, **kwargs):
    if figsize is None:
        figsize = (ncols * 6, max_n // ncols * 3)
    if ctxs is None:
        ctxs = get_grid(min(x[0].shape[0], max_n), nrows=None, ncols=ncols, figsize=figsize)
    for i, ctx in enumerate(ctxs):
        title = f'Actual: {["Not similar","Similar"][x[2][i].item()]} \n Prediction: {["Not similar","Similar"][y[2][i].item()]}'
        SiameseImage(x[0][i], x[1][i], title).show(ctx=ctx)
```


```python
learn.show_results()
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








    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_43_2.png)
    



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
imgtest = PILImage.create(files[0])
imgval = PILImage.create(files[100])
siamtest = SiameseImage(imgval, imgtest)
siamtest.show();
```


    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_45_0.png)
    



```python
res = learn.siampredict(siamtest)
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








    
![png]({{site.baseurl}}/asserts/img/2022-09-29-siamese-network-tutorial-with-fastai-update_files/2022-09-29-siamese-network-tutorial-with-fastai-update_46_2.png)
    


## Save the Model

The last step is to save the model so you can use it later.


```python
import dill
```


```python
learn.path = Path(os.getenv("MODELS"))
```


```python
learn.export("siam_catsvdogs_tutorial.pkl", pickle_module=dill)
```


```python
learn.save("siam_catsvdogs_tutorial_save")
```




    Path('/home/julius/models/models/siam_catsvdogs_tutorial_save.pth')


