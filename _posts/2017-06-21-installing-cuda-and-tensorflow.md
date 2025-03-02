---
layout: post
title: "Installing and Debugging CUDA and TensorFlow"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/spots.jpg"
tags: [CUDA, TensorFlow]
---

This tutorial aims to guide you through the process of troubleshooting issues related to your NVIDIA graphics card when working on deep learning projects.

<b>Table of Contents</b>
* TOC
{:toc}

## Hardware Requirements

Before diving in, ensure your graphics card is properly installed and compatible with CUDA:
* To find your graphics card, type `nvidia-smi`
* On Windows, you can use a tool like [Speccy](https://www.ccleaner.com/speccy).
* Verify your graphics card is listed on the [NVIDIA CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).


## Identifying Your Current Setup

Sometimes you'll get stuck somewhere in the middle of an installation and you're unsure of what installed correctly. You don't want to start from the beginning because you don't want to have multiple versions conflicting, but you don't know what you need to do next. That's why I want to start this off with some ways for you to figure out exactly where you are in the process. A good place to start is to see if TensorFlow can detect the GPUs. Here's a good one-liner for that:
```
python -c "import tensorflow as tf; print('tf version:', tf.__version__); print('Num GPU devices: ', len(tf.config.list_physical_devices('GPU')))"
```

If it says `Num GPU devices: 0` then your GPUs are not being recognized.

![png]({{site.baseurl}}/asserts/img/assets/img/zero_gpus.png)


## Compatibility

Version incompatibility is probably the biggest source of problems in getting CUDA working with TensorFlow. Let's look at how to make it all work together. I think it helps to make a table of the versions you are explicitly targeting. Here's an example:


| Software      | How to determine version | Version Targeting
| ----------- | ----------- | ----------- |
| TensorFlow      | Choose which version you want |   2.13 |
| Python      | Choose compatible version from [this chart](https://www.tensorflow.org/install/source#gpu) |   3.11 |
| CUDA  | Choose compatible version from [this chart](https://www.tensorflow.org/install/source#gpu) | 11.8 |
| cuDNN | Choose compatible version from [this chart](https://www.tensorflow.org/install/source#gpu) | 8.6 |
| NVIDIA Drivers | Choose the latest version. They are always backwards compatible | 545.23.06 |

Get these EXACT versions except for the NVIDIA driver, which you just need at least the version. Note that new CUDA versions are not backwards compatible.

You also need to decide if you want it to run on a specific conda environment or if you want things installed system-wide. For example, if you download and install cuDNN from NVIDIA's website, it will be system-wide. But if you install it using conda, it will only be for that environment. In general:
* If you want multiple projects or environments to share a single cuDNN installation and always be on the same version, a system-wide installation may be more appropriate.
* If you want to have different versions of cuDNN for different projects or if you want to ensure that a particular project always uses a specific version regardless of system-wide changes, a conda-specific installation is more appropriate.

## System Check

You'll need to check if everything on your system is working. Here's a quick summary table for the different parts you need and how to check them. For more details, see the sections below.

#### Summary Table

| Software      | How to check |
| ----------- | ----------- |
| NVIDIA Drivers      | `nvidia-smi`       |
| CUDA Toolkit   | `nvcc --version`        |
| cuDNN | ``cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`` |
| TensorFlow | `python -c "import tensorflow as tf; print(tf.__version__)"` |

Now let's go over each part in more detail.

#### GPUs and NVIDIA Drivers

type `nvidia-smi` and see if everything there is as you expect

The result should look something like this:

```
julius@julius-MS-7B09:~/git$ nvidia-smi
Tue Feb  2 15:05:43 2021       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.32.03    Driver Version: 460.32.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce RTX 208...  Off  | 00000000:0A:00.0 Off |                  N/A |
|  0%   31C    P8     8W / 260W |      5MiB / 11019MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  GeForce RTX 208...  Off  | 00000000:42:00.0  On |                  N/A |
| 41%   36C    P5    21W / 260W |    632MiB / 11016MiB |     27%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1255      G   /usr/lib/xorg/Xorg                  4MiB |
|    1   N/A  N/A      1255      G   /usr/lib/xorg/Xorg                350MiB |
|    1   N/A  N/A      1687      G   /usr/bin/gnome-shell               84MiB |
|    1   N/A  N/A     45805      G   ...gAAAAAAAAA --shared-files       95MiB |
|    1   N/A  N/A     46570      G   ...gAAAAAAAAA --shared-files       99MiB |
+-----------------------------------------------------------------------------+
```
> Note: The CUDA Version displayed isn't there isn't necessarily the version you have. It's the highest version that your driver can support.

##### Windows

To check your driver version on Windows you can also go to GEForce Experience and click on "Drivers".

#### CUDA version

You can see which version of CUDA you have installed with this:

* `cat /usr/local/cuda/version.txt`

You can see which versions of CUDA are available through `conda` with this:

* `conda search cudatoolkit`

You can also find your version by opening a command prompt and entering `nvcc -V`

```
(tf) julius@julius-MS-7B09:~/git$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```

#### cuDNN version

If you've found CUDA and you have the right version, you'll want to look for cuDNN. You'll probably be able to find it with this:

* `cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

But if you're not sure you can check with these:
* `cat $(whereis cudnn.h) | grep CUDNN_MAJOR -A 2`
* `cat $(whereis cuda)/include/cudnn.h | grep CUDNN_MAJOR -A 2`

###### Windows

If you're on Windows you should be able to find them here:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include`

You can also type this: `where cudnn*`

## Installation

There are many ways to install different components, this gives some suggestions.

#### NVIDIA Driver

Install the most recent NVIDIA Driver from [NVIDIA's website](http://www.nvidia.com/Download/index.aspx?lang=en-us)

#### CUDA Toolkit

Install the [CUDA Toolkit here](https://developer.nvidia.com/cuda-downloads). 

If you want an older version, you can [go to their archives](https://developer.nvidia.com/cuda-toolkit-archive). 

Note that on Ubuntu you can also install it like so: `sudo apt install nvidia-cuda-toolkit`. However, you won't be able to control the version, so I recommend installing it from the [website](https://developer.nvidia.com/cuda-downloads).

After you input your operation system information, you have the option of downloading either the network or local installer. Either will work. The Network Installer allows you to download only the files you need. The Local Installer is a stand-alone installer with a large initial download.

You can check that it worked by running `nvcc --version`.

Cuda is usually installed in `/usr/local/`. If you don't see it here but `nvcc --version` works, it may be installed in your environment. To find out, you can do `which nvcc`.

If you want to install it in your code environment, you can do so by specifying the particular version like so: `conda install cudatoolkit==11.8`. This doesn't have `nvcc` though.

##### Ubuntu

You should be able to install on a debian-based Linux with `sudo apt install nvidia-cuda-toolkit`.

##### Windows

After you extract CUDA on Windows, it might go to a location like `C:\Users\Julius\AppData\Local\Temp\CUDA`

This site also has a lot of good information for installing on Windows: http://blog.nitishmutha.com/tensorflow/2017/01/22/TensorFlow-with-gpu-for-windows.html


#### cuDNN

You can find the [latest version of cuDNN here](https://developer.nvidia.com/rdp/cudnn-download). Make sure to choose the appropriate cuDNN version that matches your CUDA Toolkit version. You might need a version from the [cuDNN archive](https://developer.nvidia.com/rdp/cudnn-archive).

You can also find the [release notes on old versions of cuDNN](https://docs.nvidia.com/deeplearning/cudnn/archives/index.html).

You will need to log in to download it, but it's free. Then you should see a page like this:

![image](https://github.com/jss367/jss367.github.io/assets/3067731/7e5c0170-9d74-4c04-95cd-6009d41e59c2)

You probably want something like this: `Local Installer for Ubuntu22.04 x86_64 (Deb)`

That will save a .deb file.

Now, you'll need to navigate to it in a terminal. If it downloaded to `Downloads`, that would look like this:

```
cd Downloads
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb 
```

Now we need to update our environment variables:
```
echo 'export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}' >> ~/.bashrc
source ~/.bashrc
```

If you installed from a .tgz file, you'll need to update `PATH` too.


#### Windows

The guide to install cuDNN on Windows is [here](http://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#install-windows).

You'll need to download and unzip the files. For me, they download into a folder like `C:\Users\Julius\Downloads\cudnn-10.1-windows10-x64-v7.6.5.32`

Then you have to copy files:

Copy the following files into the CUDA Toolkit directory.

Going from here:
* `C:\Users\HMISYS\Downloads\cudnn-8.0-windows7-x64-v6.0\cuda\lib\x64`

to here:
* `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0`

Copy the following files into the CUDA Toolkit directory.
* Copy <installpath>\cuda\bin\cudnn64_7.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin.
* Copy <installpath>\cuda\ include\cudnn.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\include.
* Copy <installpath>\cuda\lib\x64\cudnn.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64.

Make sure you get the version right. You can cd to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` and the use `dir` to see what versions you have (it should match what you previously saw).

After extracting the files, here's my install path:
`C:\Users\Julius\Downloads\cudnn-10.1-windows10-x64-v7.6.5.32`

For these to work you'll probably need to run as administrator

Then, from your install path, you'll want to:

`copy cuda\bin\cudnn*.dll "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin"`
`copy cuda\include\cudnn*.h "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include"`
`copy cuda\lib\x64\cudnn*.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64"`

Make sure your destination is right too:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\bin

## Tensorflow

You can install or upgrade it with: `pip install tensorflow --upgrade`

You can download the wheels from here: https://github.com/mind/wheels/releases/tag/tf1.4-gpu-cuda9


## Installing

Once you've downloaded the correct version, it will probably be in a Download folder, somewhere like `/home/julius/Downloads`

You'll need to extract it with something like this:
`tar -xzvf cudnn-10.1-linux-x64-v7.6.5.32.tgz`

Then you'll need to copy the files like so:
``` bash
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
```
Then change the permissions like so:
``` bash
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```
                                                                               
                                                                                                           
## Paths

On Unix machines, you'll need to add these to your .bashrc:
```
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
export INCLUDE=/usr/local/cuda/include
```

## Test if Tensorflow is working on the GPU

You can see all your physical devices like so:
``` python
import tensorflow as tf
tf.config.experimental.list_physical_devices()
```
and you can limit them to the GPU:
``` python
tf.config.experimental.list_physical_devices('GPU')
```
``` python
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
```


## Other Problems

If you're still stuck at the end, you can try:
```
python -c "import tensorflow as tf; print('tf version:', tf.version); tf.config.list_physical_devices()"
```
This should give you an error message that you can Google.




`python -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"`


`print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))`


Check whether devices are availble
```bash
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
```


What to do if they are not?









if it says 110 that means 11.X, so if the latest is 11.2 that's fine.




Unzip it






`<installpath>\cuda\bin\cudnn*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vx.x\bin`

If you downloaded it to Downloads, after you extract it


`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`


it will look like this

``
cp C:\Users\Julius\Downloads
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2
``

It'll extract to C:\Users\Julius\Downloads\cudnn-11.2-windows-x64-v8.1.1.33

```

`copy C:\Users\Julius\Downloads\cudnn-11.2-windows-x64-v8.1.1.33\cuda\bin\cudnn*.dll "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\bin"`


`copy C:\Users\Julius\Downloads\cudnn-11.2-windows-x64-v8.1.1.33\cuda\include\cudnn*.h "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\include"`


`copy C:\Users\Julius\Downloads\cudnn-11.2-windows-x64-v8.1.1.33\cuda\lib\x64\cudnn*.lib "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2\lib\x64"`

```

Then set your environment variables
* Do the system variables not the user ones
* This might already be done but ensure that it is


#### gcc

Make sure you have gcc:
`gcc --version`


#### Verify that there aren't conflicting drivers (Linux only)

verify you have CUDA-enabled GPU:

You should see something saying "NVIDIA" when you do:

`lspci | grep -i nvidia`

But you shouldn't see anything from:

`lsmod | grep nouveau`

If you do, you'll need to remove it
  
# Old information
  
Some things are no longer relevant to the latest version of TensorFlow, but might be helping in debugging old versions. I've move that information here.

Although tensorflow-gpu and tensorflow is a distinction of version <= 1.15, the distinction matters quite a lot here. If you do `conda create -n tf tensorflow` it will not create a GPU version, even though it installs a 2.X version of Tensorflow. You'll need to use `conda create -n tf tensorflow-gpu` to get the GPU version.

## Instructions with Anaconda

I recommend installing CUDA through Anaconda when possible.

Install Anaconda like normal. It's a little annoying on Windows because of how paths work. If you are on Windows, don't add it to the path environment in the setup window

## Installing with Pip

See instructions here: https://www.tensorflow.org/install/pip
