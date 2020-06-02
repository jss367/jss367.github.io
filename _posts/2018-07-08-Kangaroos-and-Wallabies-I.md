---
layout: post
title: "Kangaroos and Wallabies I: Preparing the Data"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/glass_ball.jpg"
tags: [Python, Computer Vision, Keras, Wildlife]
---

In this series of posts, I will show how to build an image classifier using your own dataset. I'll be using images of kangaroos and wallabies that I've taken, but these techniques should work well with any kind of images.

This is part one of a three-post series on creating your own dataset and classifying it using transfer learning.
* [Preparing the Data](https://jss367.github.io/Kangaroos-and-Wallabies-I.html)
* [Augmenting the Data](https://jss367.github.io/Kangaroos-and-Wallabies-II.html)
* [Classifying the Data](https://jss367.github.io/Kangaroos-and-Wallabies-III.html)

<b>Table of contents</b>
* TOC
{:toc}

## Introduction

First, a fun fact on kangaroos and wallabies: There isn't a clear phylogenetic boundary between the two. It's simply that the smaller ones - those where the adult male weighs less than 20 kg and has feet less than 25 centimeters - are called "wallabies", and the larger ones are called "kangaroos". In fact, some wallabies are more closely related to kangaroos than they are to other wallabies. This tammar wallaby (top) is more closely related to this red kangaroo (middle) than to the black-footed rock wallaby (bottom).

![tammar]({{site.baseurl}}/assets/img/kangwall/tammar.jpg) ![red]({{site.baseurl}}/assets/img/kangwall/red.jpg) ![bfrw]({{site.baseurl}}/assets/img/kangwall/bfrw.jpg)

OK, let's get started with the directory structure.

## Directories

To start, I have all of my kangaroo images in one folder and all my wallaby images in another. They need to be split up into training, validation, and testing sets.

We're starting with this:
```
original_data
│
└───kangaroos
│         kangaroo1.jpg
│         kangaroo2.jpg
└───wallabies
          wallaby1.jpg
          wallaby2.jpg
```
We want to move to this:
```
data
│
└───train
│   └───kangaroos
│   │     kangaroo1.jpg
│   │     kangaroo2.jpg
│   └───wallabies
│         wallaby1.jpg
│         wallaby2.jpg
└───validate
│   └───kangaroos
│   │     kangaroo3.jpg
│   │     kangaroo4.jpg
│   └───wallabies
│         wallaby3.jpg
│         wallaby4.jpg
└───test
    └───kangaroos
    │     kangaroo5.jpg
    │     kangaroo6.jpg
    └───wallabies
          wallaby5.jpg
          wallaby6.jpg```

Let's import some libraries we'll use and provide the path locations where the data are and where we want it to be.


```python
# All imports at the top
import os
from collections import Counter
import random
from shutil import copyfile
from IPython.core.debugger import set_trace
```


```python
# Locations where the data are
original_kangaroo_path = 'I:/original_data/kangaroos/'
original_wallaby_path = 'I:/original_data/wallabies/'

# Locations where we want to copy the data to
kangaroo_train_path = 'I:/data/train/kangaroos/'
kangaroo_validate_path = 'I:/data/validate/kangaroos/'
kangaroo_test_path = 'I:/data/test/kangaroos/'
wallaby_train_path = 'I:/data/train/wallabies/'
wallaby_validate_path = 'I:/data/validate/wallabies/'
wallaby_test_path = 'I:/data/test/wallabies/'
```

We'll use the [Class for Preparing Images](https://jss367.github.io/Class-for-Preparing-Images.html), which helps in presenting the data in the way we need it.


```python
%run "Class_for_Preparing_Images.py"
```

    Using TensorFlow backend.
    


```python
data_class = GatherData()
```

## Similar images

In an ideal world, all the images in a dataset would be "equal" in the sense that they're all the same quality and equally different from the others, or at least if they're unequal, the "unequalness" is randomly scattered throughout the dataset. But this isn't an ideal dataset; it's a real one. And it is not perfect.

Usually, in wildlife photography, I take a series of photos in a row to make sure I get at least one good one. This means that I have many photos taken within seconds of each other, resulting in nearly identical photos. This presents a problem for splitting the data into training, validating, and testing sets, as we can't have nearly identical images in different sets or the model could overfit on superfluous details and, because those same superfluous details are in the validate and test sets, appear to be better than it really is.

Let's take a look at two photos taken next to each other.


```python
kangaroo_files = data_class.get_filenames(original_kangaroo_path)
```


```python
similar_files = [os.path.join(original_kangaroo_path, kangaroo_files[6]), os.path.join(
    original_kangaroo_path, kangaroo_files[7])]
similar_images = [Image.open(img) for img in similar_files]
combined_images = np.hstack((np.asarray(img) for img in similar_images))
Image.fromarray(combined_images)
```




![png]({{site.baseurl}}/assets/img/2018-07-08-Preparing-folder-structure_files/2018-07-08-Preparing-folder-structure_21_0.png)



These are almost identical. We could just take one from each similar group of photos, but having nearly identical images isn't always a bad thing. In fact, it can be useful because there will be small variations in the background and specific pixel locations, but the subject is the same. That helps the model learn what is and isn't important in the image. So we're going to keep them in, we just have to confine each series of similar images to a single one of the datasets (which one doesn't actually matter, just that each series is confined to one).

To separate them, we'll have to separate the images by date, so all the images taken on one day go to one of the datasets. One way to do that would be to put all the images taken before some date in the training set, then validate and test on sets taken later. But that creates another problem. The first picture I took doesn't look much like the most recent pictures. The first photos were taken with my phone and the later ones with a real camera. 


```python
im = Image.open(os.path.join(original_kangaroo_path, kangaroo_files[0]))
```


```python
im
```




![png]({{site.baseurl}}/assets/img/2018-07-08-Preparing-folder-structure_files/2018-07-08-Preparing-folder-structure_25_0.png)



My first picture of a kangaroo, taken with my phone. Obviously, I (hope) I have improved over time. We don't want to train on a bunch of crappy images and test on good ones. So we'll have to mix it up. Let's look into the metadata to see how to do that.

## Metadata

There's a lot of information hidden in the metadata. Time, location, and equipment are often embedded in there. You can actually see my evolution in cameras in the metadata, starting with using my phone and ending with a Nikon D7200.


```python
for i in [1, 13, 100]:
    print(Image.open(os.path.join(original_kangaroo_path,
                                  kangaroo_files[i]))._getexif()[272])
```

    SCH-I545
    Canon PowerShot SX610 HS
    NIKON D7200
    

We can separate them by date, putting some dates in the training set, and others in the validate and test sets. This will keep similar images together and spread out the images taken with different cameras. We can find the capture data in the metadata by calling `_getexif()`.


```python
im._getexif()
```




    {305: 'Adobe Photoshop Lightroom Classic 7.4 (Windows)',
     306: '2018:08:22 23:06:59',
     296: 2,
     34665: 170,
     282: (240, 1),
     283: (240, 1),
     36864: b'0230',
     40961: 1}



The data is in index 36867, which is missing from the above metadata. We'll have to find and remove the images without that tag. Most photos should have much more metadata than that. Let's check another.


```python
Image.open(os.path.join(original_kangaroo_path, kangaroo_files[1]))._getexif()
```




    {296: 2,
     34665: 212,
     271: 'samsung',
     272: 'SCH-I545',
     305: 'Adobe Photoshop Lightroom Classic 7.4 (Windows)',
     306: '2018:08:22 23:07:01',
     282: (240, 1),
     283: (240, 1),
     36864: b'0230',
     37377: (6906891, 1000000),
     37378: (2275007, 1000000),
     36867: '2015:11:07 17:20:05',
     36868: '2015:11:07 17:20:05',
     37379: (280320, 65536),
     37380: (0, 10),
     37381: (228, 100),
     37383: 2,
     37384: 0,
     37385: 0,
     37386: (420, 100),
     37510: b'ASCII\x00\x00\x00METADATA-START',
     40961: 1,
     41989: 31,
     41990: 0,
     41495: 2,
     33434: (1, 120),
     33437: (22, 10),
     41729: b'\x00',
     34850: 2,
     34855: 80,
     41986: 0,
     41987: 0}



Much better. This photo was taken with a Samsung. For more information on what the different values mean, see this [post by Nicholas Armstrong](http://nicholasarmstrong.com/2010/02/exif-quick-reference/).

## Cleaning out bad data

We saw above the some of the files don't have all the required information. We'll have to remove those. It is also a good practice to look through the dataset to ensure that the images are correctly labeled. I hand-labeled this dataset, so I can confirm that it is accurate.


```python
def remove_files_without_dates(path, list_of_files):
    dates = []
    no_date = []
    for i in range(len(list_of_files)):
        im = Image.open(os.path.join(path, list_of_files[i]))
        if 36867 in im._getexif():
            # Because they're in the same format each time, we can do this which is faster than re
            dates.append(im._getexif()[
                         36867][0:4] + im._getexif()[36867][5:7] + im._getexif()[36867][8:10])
        else:
            no_date.append(list_of_files[i])
    if len(no_date):
        print("1 file was missing its date and will not be used")
    else:
        print("{} file(s) were missing dates and will not be used".format(len(no_date)))
    good_files = [file for file in list_of_files if file not in no_date]
    return (dates, good_files)
```


```python
kangaroo_file_dates, good_kangaroo_files = remove_files_without_dates(original_kangaroo_path, kangaroo_files)
```

    1 file was missing its date and will not be used
    

Only one bad one, that's good. OK, let's see how many images we have left


```python
print(len(good_kangaroo_files))
```

    2382
    

We'll have a couple thousand pictures to work with.

## Splitting the data by date

Let's build a function that finds the number of images in a day and assigns them to one of the data sets. We'll tell it how big each set should be and the function will tell us what days to include in that set.


```python
def split_days(list_of_dates, validate_set_size, test_set_size, seed=None):
    # See how many images are in each date
    counts = Counter(list_of_dates)
    validate_set_cutoff = int(sum(counts.values()) * validate_set_size)
    test_set_cutoff = int(sum(counts.values()) * test_set_size)

    # shuffle the values
    list_of_counts = list(counts.items())
    random.seed(seed)
    random.shuffle(list_of_counts)

    # Create empty data structures
    num_images_in_validate_set = 0
    num_images_in_test_set = 0
    nth_element_of_list = 0
    validate_dates = []
    test_dates = []

    # add dates to the test set until you have reached the cutoff
    for i, date_count_tuple in enumerate(list_of_counts):
        if num_images_in_validate_set < validate_set_cutoff:
            # Add the date to list of dates in this set
            validate_dates.append(date_count_tuple[0])
            # keep track of how many have been added
            num_images_in_validate_set += date_count_tuple[1]
        elif num_images_in_test_set < test_set_cutoff:
            # If the validate set is full, start on the test set
            test_dates.append(date_count_tuple[0])
            # keep track of how many have been added
            num_images_in_test_set += date_count_tuple[1]
        else:
            # both sets are full, no need to continue
            break
    return validate_dates, test_dates
```

Now we'll set the size of the train, validate, and test sets. We want to make training set as large as possible given the constraint that the variance in our validate and tests sets is low. The lower the variance, the more confidence we can have in our results. With only a few thousand images, we already don't have as much data as we would like. To keep the validate and test sets large enough to be useful, I think we should make them about 10% of our data. That leaves the other 80% for training.


```python
# Set the size of the validate and test sets
validate_set_size = 0.1
test_set_size = 0.1
```


```python
kangaroo_validate_dates, kangaroo_test_dates = split_days(kangaroo_file_dates, validate_set_size, test_set_size)
```

Now that we've found which dates go in the test set, we'll copy all our images into training and testing folders.


```python
def copy_files(original_path, train_path, validate_path, test_path, files, validate_dates, test_dates):
    data_class.make_dir_if_needed(train_path)
    data_class.make_dir_if_needed(validate_path)
    data_class.make_dir_if_needed(test_path)

    for file in files:
        im = Image.open(os.path.join(original_path, file))

        if im._getexif()[36867][0:4] + im._getexif()[36867][5:7] + im._getexif()[36867][8:10] in validate_dates:
            # move to validate set
            copyfile(os.path.join(original_path, file),
                     os.path.join(validate_path, file))
        elif im._getexif()[36867][0:4] + im._getexif()[36867][5:7] + im._getexif()[36867][8:10] in test_dates:
            # move to test set
            copyfile(os.path.join(original_path, file),
                     os.path.join(test_path, file))
        else:
            # move to train set
            copyfile(os.path.join(original_path, file),
                     os.path.join(train_path, file))
```


```python
copy_files(original_kangaroo_path, kangaroo_train_path, kangaroo_validate_path, kangaroo_test_path,
           good_kangaroo_files, kangaroo_validate_dates, kangaroo_test_dates)
```

## Repeat for Other Class

Now let's do the same thing with the wallaby images.


```python
wallaby_files = data_class.get_filenames(original_wallaby_path)
```


```python
wallaby_file_dates, good_wallaby_files = remove_files_without_dates(
    original_wallaby_path, wallaby_files)
```

    1 file was missing its date and will not be used
    


```python
wallaby_validate_dates, wallaby_test_dates = split_days(
    wallaby_file_dates, validate_set_size, test_set_size)
```


```python
copy_files(original_wallaby_path, wallaby_train_path, wallaby_validate_path, wallaby_test_path,
           good_wallaby_files, wallaby_validate_dates, wallaby_test_dates)
```


```python
print(len(good_wallaby_files))
```

    1652
    

There are more images of kangaroos than wallabies, so the datasets will be unbalanced. That won't be a problem as long as we take it into consideration when we measure the quality of the model in the third post.

## Next steps

Although we have thousands of images to work with, that isn't nearly enough to train state-of-the-art neural networks. In the next notebook, we'll look at how we can [augment our dataset with Keras](https://jss367.github.io/Kangaroos-and-Wallabies-II.html).
