---
layout: post
title: "Class for Preparing Images"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/kodiak_sunset.jpg"
tags: [Python, Computer Vision, Keras]
---


This class helps handle image data for use in machine learning. It helps to read image files from the directory and convert them into training and testing sets. It has the flexibility to return the data in any of the following forms:

* grayscale or RGB
* flattened vectors or not
* any square image size
* standardized, normalized, or not
* labels as either column vector or not ( (n,) or (n, 1) )
* number of samples either as first or last in ndarray

# Class

```python
import random
from os import listdir, makedirs
from os.path import isfile
from pathlib import Path

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import (array_to_img, img_to_array,
                                                  load_img)

"""This is specifically written for the kangaroos and wallabies dataset
TODO: Expand to more than two classes
TODO: Remove strings as enums
"""


class DataMaster(object):
    '''
    This class helps handle image data for use in machine learning.
    It helps to read image files from the directory and convert them into training and testing sets.
    It has the flexibility to return the data in any of the following forms:

    * grayscale or rgb
    * flattened vectors or not
    * any square image size
    * standardized, normalized, or not
    * labels as either column vector or not ( (n,) or (n, 1) )
    * number of samples either as first or last in ndarray
    '''

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.image_size = None
        self.num_train_images = None
        self.num_test_images = None

    def get_filenames(self, path):
        '''
        Returns list of filenames in a path
        '''
        path = Path(path)
        files = [file for file in listdir(path) if isfile(path / file)]
        return files

    def get_images(
            self,
            path,
            result_format='list of PIL images',
            new_size=0,
            grayscale=True):
        '''
        Accepts a path to a directory of images and
        returns an ndarray of shape N, H, W, c where
        N is the number of images
        H is the height of the images
        W is the width of the images\
        c is 3 if RGB and 1 if grayscale

        result can be "ndarray" for a single large ndarray,
        "list of ndarrays", or list of PIL Images (PIL.Image.Image)

        This function also allows the images to be resized, but forces square
        image_size passes size of first image only
        '''
        files = [Path(f) for f in self.get_filenames(path)]
        images = []
        for im_file in files:
            image = Image.open(path / im_file)
            if grayscale:
                image = image.convert("L")
            if new_size != 0:
                image = image.resize((new_size, new_size), Image.ANTIALIAS)
            if result_format == 'ndarray' or result_format == 'list of ndarrays':
                image = np.array(image)
            images.append(image)
        if result_format == 'ndarray':
            array_of_images = np.asarray(images)
            self.image_size = array_of_images[0].shape[0:2]
            return array_of_images
        elif result_format == 'list of ndarrays':
            self.image_size = images[0].shape[0:2]
        elif result_format == 'list of PIL images':
            self.image_size = images[0].size
        return images

    def augment_images(
            self,
            original_file,
            output_path,
            output_prefix,
            image_number,
            datagen,
            count=10):
        '''
        This function works on a single image at a time.
        It works best by enumerating a list of file names and passing the file and index.

        original_file must be the full path to the file, not just the filename

        The image_number should be the index from the enumeration e.g.:
        for index, file in enumerate(train_files):
            augment_images(os.path.join(train_path, file), output_path, str(index), datagen, count=10)
        '''
        makedirs(output_path, exist_ok=True)

        # load image to array
        image = img_to_array(load_img(original_file))

        # reshape to array rank 4
        image = image.reshape((1,) + image.shape)

        # let's create infinite flow of images
        images_flow = datagen.flow(image, batch_size=1)
        for index, new_images in enumerate(images_flow):
            if index >= count:
                break
            # we access only first image because of batch_size=1
            new_image = array_to_img(new_images[0], scale=True)

            output_filename = (output_path + output_prefix + image_number + '-' + str(index + 1) + '.jpg')

            new_image.save(output_filename)

    def train_test_sets(
        self,
        input1_training_path,
        input2_training_path,
        input1_testing_path,
        input2_testing_path,
        new_size=256,
        grayscale=False,
        num_samples_last=False,
        standardization='normalize',
        seed=42,
        verbose=False,
        y_as_column_vector=False,
        flatten=True,
    ):
        '''
        This accepts paths for the training and testing location of two paths
        To leave the images at their original size pass `new_size = 0`

        '''

        # Get an ndarray of each group of images
        # Array should be N * H * W * c
        if verbose:
            print("getting images")
        train1 = self.get_images(
            input1_training_path,
            result_format='ndarray',
            new_size=new_size,
            grayscale=grayscale)
        train2 = self.get_images(
            input2_training_path,
            result_format='ndarray',
            new_size=new_size,
            grayscale=grayscale)
        test1 = self.get_images(
            input1_testing_path,
            result_format='ndarray',
            new_size=new_size,
            grayscale=grayscale)
        test2 = self.get_images(
            input2_testing_path,
            result_format='ndarray',
            new_size=new_size,
            grayscale=grayscale)

        # make sure the image is square
        assert train1.shape[1] == train1.shape[2] == new_size
        # Now we have an array of images N * W * H * 3 or N * W * H * 1

        if flatten:
            if verbose:
                print("flattening")
            # Flatten the arrays
            if grayscale:
                flattened_size = new_size * new_size
            else:
                flattened_size = new_size * new_size * 3

            train1 = train1.reshape(train1.shape[0], flattened_size)
            train2 = train2.reshape(train2.shape[0], flattened_size)
            test1 = test1.reshape(test1.shape[0], flattened_size)
            test2 = test2.reshape(test2.shape[0], flattened_size)

        # Combine the two different inputs into a single training set
        training_images = np.concatenate((train1, train2), axis=0)
        # Do same for testing set
        testing_images = np.concatenate((test1, test2), axis=0)

        # Get the number of training and testing examples
        self.num_train_images = len(training_images)
        self.num_test_images = len(testing_images)

        # Create labels
        training_labels = np.concatenate((np.zeros(len(train1)), np.ones(len(train2))))
        testing_labels = np.concatenate((np.zeros(len(test1)), np.ones(len(test2))))

        # Zip the images and labels together so they can be shuffled together
        if verbose:
            print("zipping")
        train_zipped = list(zip(training_images, training_labels))

        if verbose:
            print("shuffling")
        # Now shuffle training images
        random.seed(seed)
        random.shuffle(train_zipped)

        self.X_train, self.y_train = zip(*train_zipped)
        self.X_test, self.y_test = testing_images, testing_labels

        # Convert tuples back to ndarrays
        self.X_train = np.asarray(self.X_train)
        self.X_test = np.asarray(self.X_test)
        self.y_train = np.asarray(self.y_train)
        self.y_test = np.asarray(self.y_test)

        if standardization == 'normalize':
            if verbose:
                print("standardizing")
            # Standardize the values
            self.X_train = (self.X_train - self.X_train.mean()) / self.X_train.std()
            # Use the train mean and standard deviation
            self.X_test = (self.X_test - self.X_train.mean()) / self.X_train.std()
        elif standardization == 'rescale':
            if verbose:
                print("standardizing")
            # Standardize the values
            self.X_train = self.X_train / 255.0
            self.X_test = self.X_test / 255.0

        if y_as_column_vector:
            # Reshape the y to matrix them n X 1 matrices
            self.y_train = self.y_train.reshape(self.y_train.shape[0], 1)
            self.y_test = self.y_test.reshape(self.y_test.shape[0], 1)

        if num_samples_last:
            # Reshape array to (L*W*c, N)
            self.X_train.shape = (self.X_train.shape[1], self.X_train.shape[0])
            self.X_test.shape = (self.X_test.shape[1], self.X_test.shape[0])
            self.y_train.shape = (self.y_train.shape[1], self.y_train.shape[0])
            self.y_test.shape = (self.y_test.shape[1], self.y_test.shape[0])

    def dataset_parameters(self):
        '''
        Returns the parameters of the dataset
        '''
        try:
            print("X_train shape: " + str(self.X_train.shape))
            print("y_train shape: " + str(self.y_train.shape))
            print("X_test shape: " + str(self.X_test.shape))
            print("y_test shape: " + str(self.y_test.shape))
            print("Number of training examples: " + str(self.num_train_images))
            print("Number of testing examples: " + str(self.num_test_images))
            print("Each image is of size: " + str(self.image_size))

        except AttributeError:
            print("Error: The data has not been input or is incorrectly configured.")


```

# Usage

## Augmenting images

`augment_images` uses the augmentation functionality of Keras, but provides more flexibility.


```python
from keras.preprocessing.image import ImageDataGenerator

my_data = DataMaster()
image_path = 'img/'
aug_path = 'aug/'
filenames = my_data.get_filenames(image_path)

datagen = ImageDataGenerator(
    rotation_range=45
)
for index, file in enumerate(filenames):
        my_data.augment_images(os.path.join(image_path, file), aug_path,
                               'augmented_image', str(index), datagen, count=2)
```

## Getting data

The function `train_test_sets` is the primary function in this class. It expects a directory in the following format:

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
└───test
    └───kangaroos
    │     kangaroo3.jpg
    │     kangaroo4.jpg
    └───wallabies
          wallaby3.jpg
          wallaby4.jpg
```
          
See [this post](https://jss367.github.io/Kangaroos-and-Wallabies-I.html) for a guide on how to set up your images like this.


```python
im1_train_path = 'data/train/im1/'
im2_train_path = 'data/train/im2/'
im1_test_path = 'data/test/im1/'
im2_test_path = 'data/test/im2/'
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
my_data.train_test_sets(im1_train_path, im2_train_path, im1_test_path, im2_test_path)
X_train = my_data.X_train_
y_train = my_data.y_train_
X_test = my_data.X_test_
y_test = my_data.y_test_
my_data.dataset_parameters()
```
