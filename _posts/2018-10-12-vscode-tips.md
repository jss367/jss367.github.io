---
layout: post
title: "VSCode Tips"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/binary.jpg"
tags: [Software]
---

This post contains some of my favorite tips and tricks for working with VSCode. For even more, I recommend [VSCode's own tips and tricks page](https://code.visualstudio.com/docs/getstarted/tips-and-tricks).

I try to keep this post up-to-date, so the information should be compatible with recent versions of VSCode. For example, a lot of [Python tools switched to extensions](https://github.com/microsoft/vscode-python/wiki/Migration-to-Python-Tools-Extensions) in 2022, and I have updated this page to reflect that.

<b>Table of Contents</b>
* TOC
{:toc}

# Basics

## Command Palette

One of the most important things you'll need to do in VSCode is to open the command palette. From there you can access all sorts of settings:

* Open the Command Palette: `CMD/CTRL + SHIFT + P`
  * You can also jump to the settings or Command Palette by clicking on the settings wheel in the bottom left corner.

## Search by Filename

You can do `CMD/CTRL + P` to open up search. 
* Just add `>` to the bar to make it the command palette.

# Hot Keys

## Autoformat

* Windows/Linux: `Alt + Shift + F`
* Mac: `Option + Shift + F`

## Clean up imports

* Windows/Linux: `Alt + Shift + O`
* Mac: `Option + Shift + O`

## Snippets

`Control + Spacebar` to open snippets
This makes it easy to do things like type `main` and get if `__name__ == '__main__':`

## Jupyter

`Shift + enter` to run through Python interactive console

# Customizations

See [my previous post for my recommended VSCode customizations](https://jss367.github.io/software-customizations.html).

# Settings 

* You can use either the User Interface (`Preferences: Open Settings (UI)`) or edit the JSON directly `Preferences: Open Settings (JSON)`. 

User settings are not stored in the project. Instead, they are at: 

* Windows: `C:\Users\Julius\AppData\Roaming\Code\User\settings.json`
* Mac: `~/Library/Application Support/Code/User/settings.json`

If you're having trouble with your Python interpreter, you can try setting `"python.defaultInterpreterPath": "/Users/julius/opt/anaconda3/envs/my_env/bin/python",`


## Applying Settings to a Single Language

You can specify that you only want some settings to apply to a single language like so:

```
    "[python]": {
        "editor.formatOnSave": true,
        "editor.formatOnPaste": false,
        "editor.tabSize": 4,
        "editor.defaultFormatter": "ms-python.python"
    },
```

# Other

## Key bindings

`Ctrl + k` to open key bindings. From there many things are just one button, such as `z` for zen mode. Double tap "Esc" to escape.

If you just hit `control + k` it brings up a list of key bindings, which you can customize.


# Extensions

You can manage your extensions by clicking on the gear logo next to the extension.

<img width="649" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/851fbfda-7d83-4bf2-b2d9-4c1ff4d970da">

This is where you can add specific details to your extensions.

<img width="510" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/b3034989-bc53-4c7a-b977-96a2de11d262">

<img width="490" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/f73dcf42-b65a-43a5-80c4-87407289043c">


## Terminal

There is a built-in terminal in VSCode. You can pull it up with `control + '&#96;'`, on either mac or windows. You can also pull it up from dragging up from the bottom of the screen.


## Debugging

To do multi-line debugging, all you have to do is hold down `shift` before you hit `return`.


## Code Completion

`Control + Space` to pull it up manually

# Workspace file

If you keep all your repos in a single folder like I do, I recommend putting your workspace file there (I call mine `workspace.code-workspace`). That way all the folders and paths are straightforward. Sometimes it will by default put them in `/Users/<username>/Library/Application Support/Code/Workspaces/<some_number>/workspace.json`. I don't use them there.

## Workspace File

You can include all your folders like this:
```
"folders": [
        {
            "path": "my_repo"
        },
        {
            "path": "my_monorepo/python_package_a"
        },
        {
            "path": "my_monorepo/python_package_b"
        },

    ],
```

You can also optionally include `"name"` if you want to change any of the names.

# launch.json

In the `launch.json` file, you can either use full paths or relative paths:

`"program": "/full/path/to/python_trainer.py"`,

or

`"program": "${file}"`,


Writing `launch.json` files is very useful. It makes it easy to run files in different configurations, such as passing different arguments. Here's the default starting place:

```json
{
    "name": "Python: Current File",
    "type": "python",
    "request": "launch",
    "program": "${file}",
    "console": "integratedTerminal"
},
```

Here's an example with arguments:

```json

{
   "name": "Python: Run New  Config",
   "type": "python",
   "request": "launch",
   "program": "/full/path/to/file.py",
   "console": "internalConsole",
   "justMyCode": false,
   "args": [
       "--config",
       "my_config",
       "--num_gpus",
       "2",

   ]
},

```

One thing I common use is `justMyCode`.
- defaults to true
- restricts debugging to only the user-written code

You can also set environmental variables like so:
It might look like this:
```
        {
            "name": "Python: My Module",
            "type": "python",
            "request": "launch",
            "module": "my_module",
            "justMyCode": true,
            "args": ["my_arg"],
            "env": {"PYTORCH_ENABLE_MPS_FALLBACK": "1"} # Add the env here
        }
```


You can also set your Python interpreter specifically for that run. You need something like the following:
* `"python": "/home/julius/miniconda3/envs/my_env/bin/python",`

#### Location

`launch.json` files can be stored in different locations. Sometimes you might have one in `git/my_repo/.vscode/launch.json`. I generally try to avoid this. Instead of one for each repo, I would put them all in `git/.vscode/launch.json`.

#### Relative paths (cwd)

Part of your command will include a reference from where to start from. One way to do that is by using `cwd`.

* `"cwd": "${workspaceFolder}"` - start from the workplace folder
  * In a multi folder workspace, you'll need to identify the folder as well. It will look something like `"${workspaceFolder:my_repo}"`
* `"cwd": "${fileDirname}"` - start from the directory of the current file. This will change depending which file you want to have open, so I only recommend using it when you're running the current file (so you'll have `"program": "${file}",` as well)
* Note that you don't always need to include `cwd`. For example, you don't need it when running a module.

#### Running a module or a program

In the `launch.json` file, you can choose to run either a `"module"` or a `"program"`.

#### Debugging subprocesses

You can also debug subprocess in VSCode. All you need to do is add `"subProcess": true,` to your `launch.json`.


## .env files
You can make `.env` files to set environment variables. Go at top of directory. Can add environment variables, python path, etc.


## Troubleshooting Python Interpreter Issues

Sometimes you have problems where Pylint seems to be using a different interpreter. Even if you select the correct interpreter and do it at the workspace level. I don't know what causes this, but here is how to fix it:

It could be caused by having something in 

"pylint.interpreter": ["/Users/julius/opt/anaconda3/envs/all2/bin/python"],

#### Testing

Sometimes the Discover Tests functionality fails, often for path issues. Remember, even if it fails you can always runs tests by doing `python -m pytest my_tests`


if discover tests fails, go to the terminal  - click on the output tab - and change it to Python Test Log

#### Other

If your linter can't see it but you can run the file

maybe it's a path that only works because of your .env file

Your linter doesn't use that. So when you run it, it will run, but not for pytest or your linter...
try to fix your linter by exporting the environment variables you want

The VSCode workspace setting `python.pythonPath` is not to be confused with the environment variable `$PYTHONPATH`.
`python.pythonPath` is the path to the Python interpreter used for debugging or running the code, while `$PYTHONPATH` is the environment variable which python uses to search for modules.
There are two different things going on here:
Where the computer looks for the python interpreter - `python.pythonPath`

And where that interpreter looks for packages - `$PYTHONPATH`


## Connecting to Remote Instances

I wrote [a guide on how to connect to remote instances](https://jss367.github.io/connecting-vscode-to-google-cloud-platform-instances.html). I recommend storing your config file at `~/.ssh/config`

## Troubleshooting

#### Environmental Variables

Sometimes environmental variables won't show up in VSCode. I've found that this can sometimes happen when VSCode is launched from the application icon. If this is happening, you can open VSCode directly from the terminal with `code .` and it should have your environmental variables. If you still don't see them, make sure they are present in your terminal.

## Syncing Across Desktops

Here's what I recommend keeping in sync between machines:

![image](https://user-images.githubusercontent.com/3067731/210435699-e71ff120-cfc3-413c-b99c-f98215d79924.png)

