---
layout: post
title: "Exploring Images with Python"
thumbnail: "assets/img/ghan_stretch.jpg"
feature-img: "assets/img/rainbow.jpg"
tags: [Python, Computer Vision]
---
Let's look at how to explore images in Python. We'll use the popular and active [Pillow](https://pillow.readthedocs.io/en/5.1.x/) fork of PIL, the [Python Imaging Library](http://www.pythonware.com/products/pil/).


```python
from PIL import Image
import numpy as np
```


```python
file_name = 'old_ghan.jpg'
```


```python
im = Image.open(file_name)
```


```python
im
```




![png]({{site.baseurl}}/assets/img/2018-06-05-Exploring%20Images%20with%20Python_files/2018-06-05-Exploring%20Images%20with%20Python_4_0.png)



## Filtering by color

PIL reads in an image using three different color filters: red, green, and blue. In the picture above, the top right corner is mostly blue. Thus when we extract the colors, there should be very little red in that corner but lots of blue. So it will be dark in the red filter and light in the blue filter.

Let's extract the colors individually and look at them.


```python
# Colors in the order of red, green, blue
r,g,b = im.split()
```


```python
r
```




![png]({{site.baseurl}}/assets/img/2018-06-05-Exploring%20Images%20with%20Python_files/2018-06-05-Exploring%20Images%20with%20Python_8_0.png)




```python
g
```




![png]({{site.baseurl}}/assets/img/2018-06-05-Exploring%20Images%20with%20Python_files/2018-06-05-Exploring%20Images%20with%20Python_9_0.png)




```python
b
```




![png]({{site.baseurl}}/assets/img/2018-06-05-Exploring%20Images%20with%20Python_files/2018-06-05-Exploring%20Images%20with%20Python_10_0.png)



## Converting to a matrix

A digital photograph is just a collection of numbers, and the great thing about a collection of numbers is that you can do data science with it. To do any data science on the images, we'll need to look at them as n-dimensional arrays (ndarrays). This will allow us to apply all the linear algebra tools that are common in data science.


```python
img_array = np.array(im)
print(img_array.shape)
```

    (791, 1841, 3)
    

The array consists of three matrices, one each for red, green, and blue. Each individual matrix is the height by the width of the image.

Now let's look at the top right corner of the image. For the red filter, it should all be low numbers, but for the blue filter it should be very high


```python
def see_top_right_corner(array, layer, amount):
    single_color = array[:,:,layer]
    corner = single_color[0:amount, -amount-1:-1]
    return corner
```

Let's look at the red filter (array 0)


```python
see_top_right_corner(img_array, 0, 8)
```




    array([[4, 3, 3, 1, 0, 0, 0, 0],
           [4, 3, 3, 1, 0, 0, 0, 0],
           [3, 3, 3, 1, 0, 0, 0, 0],
           [2, 2, 1, 1, 0, 0, 0, 0],
           [2, 2, 0, 1, 0, 0, 0, 0],
           [2, 2, 0, 0, 0, 0, 0, 0],
           [1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)



The numbers are on a scale of  0-255, so these values are very low, indicating there's not much red in the top right corner. Now let's look at blue.


```python
see_top_right_corner(img_array, 2, 8)
```




    array([[215, 215, 217, 217, 217, 217, 217, 216],
           [215, 215, 217, 217, 217, 217, 217, 216],
           [215, 215, 217, 217, 217, 217, 217, 216],
           [214, 214, 217, 217, 217, 217, 217, 216],
           [214, 216, 216, 217, 217, 217, 217, 217],
           [214, 216, 216, 216, 217, 217, 217, 217],
           [213, 216, 216, 216, 216, 217, 217, 217],
           [215, 216, 216, 216, 216, 217, 217, 217]], dtype=uint8)



## Greyscale

Although the individual filters are grey, using just one of them generally isn't the best way to convert to greyscale. Pillow (and other image libraries, like [OpenCV](https://opencv.org/)) has a way to combine the different individual colors into a greyscale that most closely matches what you would expect.


```python
im.convert("L")
```




![png]({{site.baseurl}}/assets/img/2018-06-05-Exploring%20Images%20with%20Python_files/2018-06-05-Exploring%20Images%20with%20Python_23_0.png)



## Resizing

Pillow can also resize images. It's recommended that you pass the `Image.ANTIALIAS` to the call.


```python
# Preserve the original dimensions
h_over_w = im.size[1]/im.size[0]
```


```python
new_width = 500
```


```python
im_resize = im.resize((new_width, int(new_width*h_over_w)), Image.ANTIALIAS)
im_resize
```




![png]({{site.baseurl}}/assets/img/2018-06-05-Exploring%20Images%20with%20Python_files/2018-06-05-Exploring%20Images%20with%20Python_28_0.png)



## Metadata

You can also look at metadata using Pillow.


```python
im._getexif()
```




    {271: 'NIKON CORPORATION',
     272: 'NIKON D7200',
     282: (240, 1),
     283: (240, 1),
     296: 2,
     305: 'Adobe Photoshop Lightroom 6.12 (Windows)',
     306: '2017:10:16 13:48:45',
     33434: (1, 8),
     33437: (22, 1),
     34665: 218,
     34850: 1,
     34855: 100,
     34864: 2,
     36864: b'0230',
     36867: '2017:03:31 06:41:30',
     36868: '2017:03:31 06:41:30',
     37377: (3, 1),
     37378: (8918863, 1000000),
     37380: (-12, 6),
     37381: (10, 10),
     37383: 5,
     37384: 0,
     37385: 16,
     37386: (240, 10),
     37521: '0411',
     37522: '0411',
     40961: 1,
     41486: (83841555, 32768),
     41487: (83841555, 32768),
     41488: 3,
     41495: 2,
     41728: b'\x03',
     41729: b'\x01',
     41730: b'\x02\x00\x02\x00\x00\x01\x01\x02',
     41985: 0,
     41986: 1,
     41987: 0,
     41988: (1, 1),
     41989: 36,
     41990: 0,
     41991: 0,
     41992: 0,
     41993: 0,
     41994: 0,
     41996: 0,
     42033: '2529332',
     42034: ((240, 10), (240, 10), (14, 10), (14, 10)),
     42036: '24.0 mm f/1.4'}



It's not particularly intuitive. For example, the creation data is number 36867. To see the meanings of these values, see the bottom of this [webpage by Nicholas Armstrong](http://nicholasarmstrong.com/2010/02/exif-quick-reference/).


```python
im._getexif()[36867]
```




    '2017:03:31 06:41:30'


