Saving and loading neural networks is always a little tricky. The best way to do it depends on what exactly you're trying to do. Do you want to continue training the model? If so, you'll need to save the optimizer state. If you just want to run it for inference, you might not need this. It also gets more complicated with custom functions. In this post I'll walk through how to save a FastAI model and then load it again for inference.


```python
from fastai.vision.all import *
```


```python
path = untar_data(URLs.PETS)
```


```python
files = get_image_files(path/"images")
```


```python
def label_func(f):
    return f[0].isupper()
```


```python
dls = ImageDataLoaders.from_name_func(path, files, label_func, item_tfms=Resize(224))
```

    Due to IPython and Windows limitation, python multiprocessing isn't available now.
    So `number_workers` is changed to 0 to avoid getting stuck
    


```python
# if you're on Windows, you need to add this
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
```


```python
learn = cnn_learner(dls, resnet34, metrics=error_rate)
learn.fine_tune(1)
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
    <tr>
      <td>0</td>
      <td>1.076228</td>
      <td>0.875705</td>
      <td>0.309091</td>
      <td>00:10</td>
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
      <td>0.258873</td>
      <td>0.541734</td>
      <td>0.163636</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>



```python
learn.show_results()
```






    
![png](2022-02-06-saving-and-loading-models-in-fastai_files/2022-02-06-saving-and-loading-models-in-fastai_8_1.png)
    


That looks good. Now let's save it.

# Saving

## learn.save

There are two options for saving models in FastAI, `learn.save` and `learn.export`. `learn.save` saves the model and, by default, also saves the optimizer state. This is what you want to do if you want to resume training.

If you used `learn.save`, you'll need to use `learner.load` to load it. You'll need all of your functions from the previous notebook again.

## learn.export

If you're going to save it for interence, I recommend using `.export()`. You should also make sure the model is going to save in the right place by setting `learn.path`.


```python
learn.path = Path(os.getenv('MODELS'))
```

If you have custom functions as we do above with `label_func`, you'll need to use the `dill` module to pickle your learner. If you use the default one, it will save the function names but not the content.


```python
import dill
```


```python
learn.export('catsvdogs.pkl', pickle_module=dill)
```

# Loading

Nothing in this section will use any of the code above. This could be done in a completely separate file or notebook.


```python
import dill
from fastai.tabular.all import *
```


```python
learn = load_learner(Path(os.getenv('MODELS')) / 'catsvdogs.pkl', pickle_module=dill)
```


```python
test = np.zeros((100, 100))
```


```python
learn.predict(test)
```








    ('False', tensor(0), tensor([0.6608, 0.3392]))


