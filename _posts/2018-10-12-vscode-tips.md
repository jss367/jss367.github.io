---
layout: post
title: "VSCode Tips"
tags: [Software]
---

This post contains some of my favorite tips and tricks for working with VSCode. For even more, VSCode has [their own tips and tricks page](https://code.visualstudio.com/docs/getstarted/tips-and-tricks) that I recommend.

<b>Table of contents</b>
* TOC
{:toc}

# Hot Keys

## Command Palette

One of the most important things to be able to do in VSCode is to open the command palette. From there you can access all sorts of settings:

* Open the Command Palette: `CMD/CTRL + SHIFT + P`
  * You can use either the User Interface (`Preferences: Open Settings (UI)`) or edit the JSON directly `Preferences: Open Settings (JSON)`. 
  * You can also jump to the settings or Command Palette by clicking on the settings wheel in the bottom left corner.

## Search

You can do `CMD/CTRL + P` to open up search. 
* Just add `>` to the bar to make it the command palette.


## Autoformat

Windows/Linux: `Alt + Shift + F`
Mac: `Option + Shift + F`

## Clean up imports

Windows/Linux: `Alt + Shift + O`
Mac: `Option + Shift + O`

## Snippets

`Control + Spacebar` to open snippets
This makes it easy to do things like type `main` and get if `__name__ == 'main':`

## Jupyter

`Shift + enter` to run through Python interactive console

# Customizations

See [the previous post for my recommended VSCode customizations](https://jss367.github.io/software-customizations.html).


# Settings 

User settings are not stored in the project. Instead, they are at: C:\Users\Julius\AppData\Roaming\Code\User\settings.json

In the `launch.json` file, you can either use full paths or relative paths:

"program": "/full/path/to/python_trainer.py",
            
or 

"program": "${file}",

## Don't write .pyc files

`PYTHONDONTWRITEBYTECODE=1`

# Other

## Key bindings

`Ctrl + k` to open key bindings. From there many things are just one button, such as `z` for zen mode.

If you just hit `control + k` it brings up a list of key bindings, which you can customize.

Zen mode: `Ctrl+k z`
(So hold control and hit "k", then let go of both and hit "z"). Double tap "Esc" to escape.






## Extensions

#### Extensions to try

https://marketplace.visualstudio.com/items?itemName=jithurjacob.nbpreviewer

https://github.com/hediet/vscode-debug-visualizer/tree/master/extension

## Terminal

The built-in terminal is a wonderful idea. You can pull it up with control + '&#96;', on either mac or windows. You can also pull it up from dragging up from the bottom of the screen.


## Debugging

To do multi-line debugging, all you have to do is hold down `shift` before you hit `return`.


## Code Completion

Control + Space to pull it up manually

## Shortcuts

"cwd": "${workspaceFolder}",

or

"cwd": "${fileDirname}"


## launch.json

Writing launch.json files is very useful. It makes it easy to run files in different configurations, such as passing different arguments. Here's the default starting place:

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


`justMyCode` is useful
- defaults to true
- restricts debugging to only the user-written code


#### Running a module or a program

In the `launch.json` file, you can choose to run either a `"module"` or a `"program"`.

#### Debugging subprocesses

You can also debug subprocess in VSCode. All you need to do is add `"subProcess": true,` to your `launch.json`.

## .env files ##
Go at top of directory. Can add environment variables, python path, etc.

- don't use quotes in variables names here


## Other

if your linter can't see it but you can run the file

maybe it's a path that only works because of your .env file

Your linter doesn't use that. So when you run it, it will run, but not for pytest or your linter...
try to fix your linter by exporting the environment variables you want

As far as I understand it, the VSCode workspace setting python.pythonPath is not to be confused with the environment variable $PYTHONPATH.
python.pythonPath is the path to the python interpreter used for debugging or running the code, while $PYTHONPATH is the environment variable which python uses to search for modules.
There are two different things going on here:
Where the computer looks for the python interpreter - python.pythonPath

And where that inpreter looks for packages - $PYTHONPATH

```
Host host101
  HostName juliushost101.localnetwork.internalnetwork.tld
  User root
  Port 2222
  IdentityFile ~/.ssh/id_rsa2 # you might have to do this
```
test it out like this: `ssh -F ~/config localhost`
You will need sshd in the container for this to work?

config can be
```
"program": "${file}"or"program": "/full/path/to/my/file.py"
"console": "integratedTerminal"or"console": "internalConsole"
```



command + / to multiline comment
https://code.visualstudio.com/shortcuts/keyboard-shortcuts-windows.pdf


## Testing

Sometimes the Discover Tests functionality fails, often for path issues. Remember, even if it fails you can always runs tests by doing `python -m pytest my_tests`


if discover tests fails, go to the terminal  - click on the output tab - and change it to Python Test Log


## Troubleshooting

#### Environmental Variables

Sometimes environmental variables won't show up in VSCode. I've found that this can sometimes happen whe VSCode is launch from the application icon. If this is happening, you can open VSCode directly from the terminal with `code .` and it should have your environmental variables. If you still don't see them, make sure they are present in your terminal.


## To Try
* shift cmd M
* F8 or Shift F8 to cycle between errors
