---
layout: post
title: "More Siamese Network Evaluation Fails with FastAI"
description: "This tutorial describes how not to evaluate siamese networks with the FastAI library"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/platypus4.jpg"
tags: [FastAI, Neural Networks, Python]
---

I had too many failures for one post, so this describe **even more** ways not to evaluate models with FastAI.


```python
import ast
import dill
from fastai.vision.all import *
```

Get all the pairs from the database


```python
learner = load_learner(Path(os.getenv('MODELS')) / 'siamese_catsvdogs.pkl', cpu=False, pickle_module=dill)
```


```python
df = pd.read_csv('siamese_data.csv', index_col=0)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>items</th>
      <th>label</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>['/home/julius/.fastai/data/oxford-iiit-pet/images/keeshond_13.jpg', '/home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_184.jpg']</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>['/home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_175.jpg', '/home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_24.jpg']</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>['/home/julius/.fastai/data/oxford-iiit-pet/images/german_shorthaired_145.jpg', '/home/julius/.fastai/data/oxford-iiit-pet/images/german_shorthaired_147.jpg']</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>['/home/julius/.fastai/data/oxford-iiit-pet/images/beagle_17.jpg', '/home/julius/.fastai/data/oxford-iiit-pet/images/american_bulldog_110.jpg']</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>['/home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_11.jpg', '/home/julius/.fastai/data/oxford-iiit-pet/images/Persian_5.jpg']</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.iloc[0]
```




    items       ['/home/julius/.fastai/data/oxford-iiit-pet/images/keeshond_13.jpg', '/home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_184.jpg']
    label                                                                                                                                             False
    is_valid                                                                                                                                          False
    Name: 0, dtype: object



In theory you should be able to do `learner.predict(pair_df.iloc[0])`.
This works for simple networks but fails in this case.
Instead, you can just do the first two lines of that function:
```
dl = self.dls.test_dl([item], rm_type_tfms=rm_type_tfms, num_workers=0)
inp, preds, _, dec_preds = self.get_preds(dl=dl, with_input=True, with_decoded=True)
```


```python
dl = learner.dls.test_dl(df)
```


```python
try:
    learner.get_preds(dl=dl, with_input=True)
except (IsADirectoryError, FileNotFoundError) as err:
    print(err)
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





<div>
  <progress value='0' class='' max='119' style='width:300px; height:20px; vertical-align: middle;'></progress>
  0.00% [0/119 00:00&lt;?]
</div>



    Caught FileNotFoundError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
        data = fetcher.fetch(index)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 39, in fetch
        data = next(self.dataset_iter)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 140, in create_batches
        yield from map(self.do_batch, self.chunkify(res))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/basics.py", line 230, in chunked
        res = list(itertools.islice(it, chunk_sz))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 155, in do_item
        try: return self.after_item(self.create_item(s))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 162, in create_item
        if self.indexed: return self.dataset[s or 0]
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in __getitem__
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in <listcomp>
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 509, in __getitem__
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 455, in _after_item
        return self.tfms(o)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 299, in __call__
        return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 234, in compose_tfms
        x = f(x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 117, in __call__
        return self._call("encodes", x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 132, in _call
        return self._do_call(getattr(self, fn), x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 139, in _do_call
        return retain_type(f(x, **kwargs), x, ret)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/dispatch.py", line 161, in __call__
        return f(*args, **kwargs)
      File "/tmp/ipykernel_15574/1954726133.py", line 235, in create
      File "/tmp/ipykernel_15574/1954726133.py", line -1, in <genexpr>
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 185, in create
        return cls(load_image(fn, **merge(cls._open_args, kwargs)))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 151, in load_image
        im = Image.open(fn)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3046, in open
        fp = builtins.open(filename, "rb")
    FileNotFoundError: [Errno 2] No such file or directory: '['
    
    

Converting to a tuple also doesn't help.


```python
df['items'] = df['items'].apply(ast.literal_eval)
```


```python
dl = learner.dls.test_dl(df)
```


```python
try:
    learner.get_preds(dl=dl, with_input=True)
except (IsADirectoryError, FileNotFoundError, AttributeError) as err:
    print(err)
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





<div>
  <progress value='0' class='' max='119' style='width:300px; height:20px; vertical-align: middle;'></progress>
  0.00% [0/119 00:00&lt;?]
</div>



    Caught AttributeError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3050, in open
        fp.seek(0)
    AttributeError: 'list_iterator' object has no attribute 'seek'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
        data = fetcher.fetch(index)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 39, in fetch
        data = next(self.dataset_iter)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 140, in create_batches
        yield from map(self.do_batch, self.chunkify(res))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/basics.py", line 230, in chunked
        res = list(itertools.islice(it, chunk_sz))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 155, in do_item
        try: return self.after_item(self.create_item(s))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 162, in create_item
        if self.indexed: return self.dataset[s or 0]
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in __getitem__
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in <listcomp>
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 509, in __getitem__
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 455, in _after_item
        return self.tfms(o)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 299, in __call__
        return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 234, in compose_tfms
        x = f(x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 117, in __call__
        return self._call("encodes", x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 132, in _call
        return self._do_call(getattr(self, fn), x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 139, in _do_call
        return retain_type(f(x, **kwargs), x, ret)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/dispatch.py", line 161, in __call__
        return f(*args, **kwargs)
      File "/tmp/ipykernel_15574/1954726133.py", line 235, in create
      File "/tmp/ipykernel_15574/1954726133.py", line -1, in <genexpr>
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 185, in create
        return cls(load_image(fn, **merge(cls._open_args, kwargs)))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 151, in load_image
        im = Image.open(fn)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3052, in open
        fp = io.BytesIO(fp.read())
    AttributeError: 'list_iterator' object has no attribute 'read'
    
    


```python
df['items'] = df['items'].apply(lambda x: tuple(x))
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>items</th>
      <th>label</th>
      <th>is_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(/home/julius/.fastai/data/oxford-iiit-pet/images/keeshond_13.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_184.jpg)</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(/home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_175.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Russian_Blue_24.jpg)</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(/home/julius/.fastai/data/oxford-iiit-pet/images/german_shorthaired_145.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/german_shorthaired_147.jpg)</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(/home/julius/.fastai/data/oxford-iiit-pet/images/beagle_17.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/american_bulldog_110.jpg)</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(/home/julius/.fastai/data/oxford-iiit-pet/images/leonberger_11.jpg, /home/julius/.fastai/data/oxford-iiit-pet/images/Persian_5.jpg)</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
dl = learner.dls.test_dl(df)
```


```python
try:
    learner.get_preds(dl=dl, with_input=True)
except (IsADirectoryError, FileNotFoundError, AttributeError) as err:
    print(err)
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





<div>
  <progress value='0' class='' max='119' style='width:300px; height:20px; vertical-align: middle;'></progress>
  0.00% [0/119 00:00&lt;?]
</div>



    Caught IsADirectoryError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
        data = fetcher.fetch(index)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 39, in fetch
        data = next(self.dataset_iter)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 140, in create_batches
        yield from map(self.do_batch, self.chunkify(res))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/basics.py", line 230, in chunked
        res = list(itertools.islice(it, chunk_sz))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 155, in do_item
        try: return self.after_item(self.create_item(s))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/load.py", line 162, in create_item
        if self.indexed: return self.dataset[s or 0]
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in __getitem__
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 557, in <listcomp>
        res = tuple([tl[it] for tl in self.tls])
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 509, in __getitem__
        return self._after_item(res) if is_indexer(idx) else res.map(self._after_item)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/data/core.py", line 455, in _after_item
        return self.tfms(o)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 299, in __call__
        return compose_tfms(o, tfms=self.fs, split_idx=self.split_idx)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 234, in compose_tfms
        x = f(x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 117, in __call__
        return self._call("encodes", x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 132, in _call
        return self._do_call(getattr(self, fn), x, **kwargs)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 141, in _do_call
        res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 141, in <genexpr>
        res = tuple(self._do_call(f, x_, **kwargs) for x_ in x)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/transform.py", line 139, in _do_call
        return retain_type(f(x, **kwargs), x, ret)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastcore/dispatch.py", line 161, in __call__
        return f(*args, **kwargs)
      File "/tmp/ipykernel_15574/1954726133.py", line 235, in create
      File "/tmp/ipykernel_15574/1954726133.py", line -1, in <genexpr>
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 185, in create
        return cls(load_image(fn, **merge(cls._open_args, kwargs)))
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/fastai/vision/core.py", line 151, in load_image
        im = Image.open(fn)
      File "/home/julius/miniconda3/envs/pt2/lib/python3.10/site-packages/PIL/Image.py", line 3046, in open
        fp = builtins.open(filename, "rb")
    IsADirectoryError: [Errno 21] Is a directory: '/'
    
    


```python

```
