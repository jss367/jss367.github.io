---
layout: post
title: "VSCode Tips"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/binary.jpg"
tags: [Software]
---

This post contains some of my favorite tips and tricks for working with VSCode. For even more, VSCode has [their own tips and tricks page](https://code.visualstudio.com/docs/getstarted/tips-and-tricks) that I recommend. Note that I try to keep this up-to-date, so the information should be good with roughly the latest version of VSCode. If something breaks my VSCode workflow, I update this page.

For example, a lot of [Python tools switched to extensions](https://github.com/microsoft/vscode-python/wiki/Migration-to-Python-Tools-Extensions) in 2022, and I have updated this page to include that.

<b>Table of Contents</b>
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

See [the previous post for my recommended VSCode customizations](https://jss367.github.io/software-customizations.html).

# Settings 

User settings are not stored in the project. Instead, they are at: 

* Windows: `C:\Users\Julius\AppData\Roaming\Code\User\settings.json`
* Mac: `~/Library/Application Support/Code/User/settings.json`

In the `launch.json` file, you can either use full paths or relative paths:

`"program": "/full/path/to/python_trainer.py"`,

or

`"program": "${file}"`,

If you're having trouble with your Python interpreter, you can try setting `"python.defaultInterpreterPath": "/Users/julius/opt/anaconda3/envs/my_env/bin/python",`

## Don't write .pyc files

`PYTHONDONTWRITEBYTECODE=1`


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

`Ctrl + k` to open key bindings. From there many things are just one button, such as `z` for zen mode.

If you just hit `control + k` it brings up a list of key bindings, which you can customize.

Zen mode: `Ctrl+k z`
(So hold control and hit "k", then let go of both and hit "z"). Double tap "Esc" to escape.






## Extensions

You can manage your extensions by clicking on the gear logo next to the extension.

<img width="649" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/851fbfda-7d83-4bf2-b2d9-4c1ff4d970da">

This is where you can add specific details to your extensions.

<img width="510" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/b3034989-bc53-4c7a-b977-96a2de11d262">

<img width="490" alt="image" src="https://github.com/jss367/jss367.github.io/assets/3067731/f73dcf42-b65a-43a5-80c4-87407289043c">

#### Extensions to try

https://marketplace.visualstudio.com/items?itemName=jithurjacob.nbpreviewer

https://github.com/hediet/vscode-debug-visualizer/tree/master/extension

## Terminal

The built-in terminal is a wonderful idea. You can pull it up with control + '&#96;', on either mac or windows. You can also pull it up from dragging up from the bottom of the screen.


## Debugging

To do multi-line debugging, all you have to do is hold down `shift` before you hit `return`.


## Code Completion

Control + Space to pull it up manually

## Workspace file

If you keep all your repos in a `git` folder like I do, I recommend putting your workspace file there (I call mine `workspace.code-workspace`). That way all the folders and paths are straightforward. Sometimes it will by default put them in `/Users/<username>/Library/Application Support/Code/Workspaces/<some_number>/workspace.json`. I don't use them there.

### Workspace File

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

## launch.json

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


`justMyCode` is useful
- defaults to true
- restricts debugging to only the user-written code


You can also set your Python interpreter specifically for that run. You need something like the following:
* `"python": "/home/julius/miniconda3/envs/my_env/bin/python",`

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

#### Location

`launch.json` files can be stored in different locations. Sometimes you might have one in `git/my_repo/.vscode/launch.json`. I generally try to avoid this. Instead of one for each repo, I would put them all in `git/.vscode/launch.json`.

## .env files
Go at top of directory. Can add environment variables, python path, etc.

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

- don't use quotes in variables names here


## Pylint Wrong Interpreter

Sometimes you have problems where Pylint seems to be using a different interpreter. Even if you select the correct interpreter and do it at the workspace level. I don't know what causes this, but here is how to fix it:

It could be caused by having something in 

"pylint.interpreter": ["/Users/julius/opt/anaconda3/envs/all2/bin/python"],

## Other

if your linter can't see it but you can run the file

maybe it's a path that only works because of your .env file

Your linter doesn't use that. So when you run it, it will run, but not for pytest or your linter...
try to fix your linter by exporting the environment variables you want

As far as I understand it, the VSCode workspace setting `python.pythonPath` is not to be confused with the environment variable `$PYTHONPATH`.
`python.pythonPath` is the path to the Python interpreter used for debugging or running the code, while `$PYTHONPATH` is the environment variable which python uses to search for modules.
There are two different things going on here:
Where the computer looks for the python interpreter - `python.pythonPath`

And where that inpreter looks for packages - `$PYTHONPATH`

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

## Connecting to Remote Instances

I wrote [a guide on how to connect to remote instances](https://jss367.github.io/connecting-vscode-to-google-cloud-platform-instances.html). I recommend storing your config file at `~/.ssh/config`

## Troubleshooting

#### Environmental Variables

Sometimes environmental variables won't show up in VSCode. I've found that this can sometimes happen whe VSCode is launch from the application icon. If this is happening, you can open VSCode directly from the terminal with `code .` and it should have your environmental variables. If you still don't see them, make sure they are present in your terminal.


## To Try
* shift cmd M
* F8 or Shift F8 to cycle between errors

## Syncing Across Desktops

Here's what I recommend keeping in sync between machines:

![image](https://user-images.githubusercontent.com/3067731/210435699-e71ff120-cfc3-413c-b99c-f98215d79924.png)

