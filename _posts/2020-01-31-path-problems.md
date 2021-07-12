---
layout: post
title: "Path Problems"
description: "A guide to some of the path problems you may face on various operating systems"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/dark_path.jpg"
tags: [Linux, Python, Windows]
---

Path problems are some of the most common and annoying problems machine learning engineers face, especially when frequently switching between operating systems. There are so many different issues you could have. In this post, I'll try to provide some background on possible issues and how to resolve them.

<b>Table of contents</b>
* TOC
{:toc}

# System Path

The first step is being able to find out what's on your path. You can do this by looking at the `$PATH` environmental variable. You can do this from the command line, but the exact command depends on which operation system you're using.

## Unix Systems

If you're using either a Mac or Linux, including Windows Subsystem for Linux, it's as simple as:

`echo $PATH`

## Windows

In Windows, it's not as simple to view your environmental variables because it depends on what terminal emulator you're using. If you are using [ConEmu](https://conemu.github.io/) or [Cmder](https://cmder.net/), you can

`echo %PATH%`

![png]({{site.baseurl}}/assets/img/windows_path.png)

The resulting text can be hard to read so to make it easier you can: `echo %PATH:;=&echo.%`

![png]({{site.baseurl}}/assets/img/windows_path_simple.png)

However, on Windows, it depends on what shell you're using. If you're using [Windows PowerShell](https://docs.microsoft.com/en-us/powershell/scripting/overview), you'll have to:

`echo $Env:PATH`

Windows PowerShell is the default terminal in VSCode, so this is what you'll need to use there as well.

## Adding to Your Path

If you want to temporarily add to your path:
```
set PATH="%PATH%;C:\path\to\directory\"
set PATH="%PATH%;C:\Users\Julius\Documents\GitHub\DataManager"
```

If you want to permanently add to your path:
`setx path "%PATH%;C:\path\to\directory\"`

# Python Path

On Windows, if you echo `$PYTHONPATH` you get nothing, but from inside Python you can `print(sys.path)` to see:
```
['c:\\Users\\Julius\\Documents\\GitHub\\ResNetFromScratch', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\python37.zip', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\DLLs', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\win32\\lib', 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\Pythonwin']
```


## Adding to your PYTHONPATH
```
Just go to system properties, environment variables, new...
Environment variables:
Hit windows key
type environment
Click on Environment Variables in the Advanced tab.
c:\\Users\\Julius\\Documents\\GitHub\\DataManager
- for some reason this doesn't always add to the python path though
```
Let's test this out. As we saw above, you can test out your PYTHONPATH directly from the command line:

`python -c "import sys; print(sys.path)"`


Remember that your paths are hierarchical on Windows

If you have `C:\Users\Julius\AppData\Local\Microsoft\WindowsApps` it will try then then tell you to go to the Windows store. But you can move the one you want up in the list.

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
If you change the environment variables you'll need to restart VSCode or Jupyter Notebooks. Once you restart you'll see the addition in `sys.path`


## Finding your Python Interpreter

From within Python, you can do:

``` python
import sys
sys.executable
```


## Jupyter Notebook problems

Let's say you set you can variable and can see it in `PYTHONPATH`.


you `import sys; print(sys.version)` from a command line and it's there, but when you run the same command in a Jupyter Notebook it's not there. That's because Jupyter has its own `PATH` variable called `JUPYTER_PATH`.

If you need to quickly add to it, you can start a notebook with:

``` python
import sys
sys.path.append('C:\\Users\\Julius\\Documents\\GitHub\\cv_dataclass\\src')
```

## Other

If you want to know more about how the Python `sys.path` is initialized, there's a [detailed StackOverflow answer](https://stackoverflow.com/questions/897792/where-is-pythons-sys-path-initialized-from).
