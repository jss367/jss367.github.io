---
layout: post
title: "Path problems"
description: "A guide to some of the path problems you may face on various operating systems"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/rainbow.jpg"
tags: [Linux, Windows]
---

Path problems are some of the most annoying and common problems when developing software, especially if you frequently switch between operating systems. There are so many different issues you could have. I'll try to explain a few and provide some background in the hope that that helps as well.

## Windows

Your path is stored as an environmental variable. Unfortunately, in Windows it's not as simple as I would like to view your environmental variables because it depends on what terminal emulator you're using. If you are on [ConEmu](https://conemu.github.io/) or [Cmder](https://cmder.net/), you can

`echo %PATH%`

This can be hard to read so to make it easier you can: `echo %PATH:;=&echo.%`

However, if you're using Windows PowerShell, you'll have to:

`echo $Env:PATH`

On a Mac/Linux, you would be able to 

`echo $PATH`

This also works on Windows Subsystem for Linux

## VSCode

There is an integrated terminal in VSCode. You can set it but the default it Windows Powershell, so you'll have to

`echo $Env:PATH`


## Adding to path

If you want to temporarily add to path:
```
set PATH="%PATH%;C:\path\to\directory\"
set PATH="%PATH%;C:\Users\Julius\Documents\GitHub\DataManager"
```

If you want to permanently add to path:
`setx path "%PATH%;C:\path\to\directory\"`




On windows, if you echo $PYTHONPATH you get nothing, but if you `print(sys.path)` you see:
```
['c:\\Users\\Julius\\Documents\\GitHub\\ResNetFromScratch', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\python37.zip', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\DLLs', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\Pythonwin']
```

## Print Python from command line

`python -c 'print("hello")'`

But on Windows you'll need double quotes

`python -c "print('hello')"`

You can even print out an entire machine learning model (It may download the first time if you don't already have the :

TF2.X version:

`python -c "from tensorflow.keras.applications.vgg16 import VGG16; print(VGG16().summary()"`

keras version:

`python -c "from keras.applications.vgg16 import VGG16; print(VGG16().summary()"`



detailed answer here: https://stackoverflow.com/questions/897792/where-is-pythons-sys-path-initialized-from


## Adding to your PYTHONPATH

Just go to system properties, environment variables, new...
Environment variables:
Hit windows key
type environment
Click on Environment Variables in the Advanced tab.
c:\\Users\\Julius\\Documents\\GitHub\\DataManager
- for some reason this doesn't always add to the python path though

Let's test this out. As we saw above, you can test out your PYTHONPATH directly from the command line:

`python -c "import sys; print(sys.path)"`


path problems

remember that your paths are hierarchical on windows

I you have `C:\Users\Julius\AppData\Local\Microsoft\WindowsApps` it will try then then tell you to go to the windows store. But you can move the one you want up in the list.

You can have this problem where it works if you are debugging but not if you're not debugging.


add BOTH conda and python to your path

open conda prompt then `where conda`

then `where python`

should say
`C:\Users\Julius\anaconda3\python.exe`

so use `C:\Users\Julius\anaconda3`




I use the User variables for this



Right-click anywhere in the editor window and select Run Python File in Terminal (which saves the file automatically):

- doesn't use selected interpreter





On a brand new Mac your path is:

```
julius@Juliuss-MacBook-Pro ~ % echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```



on Windows can also

```
echo $PYTHONPATH
set PYTHONPATH=C:\\Users\Julius\Documents\GitHub\facv\src\;C:\\Users\Julius\Documents\GitHub\fastai\fastai
python -c "import sys; print(sys.path)"
```

but in Windows it's `echo $PYTHONPATH$`




Note:
If you change the environment variables you'll need to restart vscode or jupyter notebooks
once you restart you'll see it in `sys.path`



## Jupyter Notebook problems

Let's say you set you can variable and can see it in PYTHONPATH.


you `import sys; print(sys.version)` from a command line and it's there, but when you run the same command in a Jupyter Notebook it's not there. That's because Jupyter has its own `PATH` variable called `JUPYTER_PATH`.

If you need to quickly add to it, you can start a notebook with:

``` python
import sys
sys.path.append('C:\\Users\\Julius\\Documents\\GitHub\\cv_dataclass\\src')
```
