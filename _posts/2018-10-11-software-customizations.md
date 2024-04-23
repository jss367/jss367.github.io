---
layout: post
title: "Software Customizations"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/car.jpg"
tags: [Linux, Mac, Software, Windows]
---

This post details the customizations I have made to the [software I use](https://jss367.github.io/software-i-use.html). I thought it might be helpful to others.

<b>Table of Contents</b>
* TOC
{:toc}

## Mac General

Here are some things I like to change on Macs and here's how I do it:

* The cursor doesn't jump from word to word by default (at least, with oh-my-zsh)
  * To fix this, you need to add the following to your .profile:
```
bindkey "\e\e[D" backward-word
bindkey "\e\e[C" forward-word
```

In general, I want the control button to do much more:
* ctrl + w to close a web page
* ctrl + arrow key to jump to next word, both in terminal and in text editor in browser.


I do this to make it easier to navigate in terminals: https://stackoverflow.com/questions/6205157/how-to-set-keyboard-shortcuts-to-jump-to-beginning-end-of-line/22312856#22312856
* also add ctrl+z for undo (so there are two ways to undo)


### Jumping around text

You'll find that whether you can jump around text depends on a specific application. For example, if you've done the above, you can use `control + shift` to highlight entire words in Gmail, but not Outlook. The solution is to stop using Outlook.


## Sublime Text

#### External Packages

Sublime Text has a great ecosystem of packages and plugins. First you need to install the package manager:
* `cmd` + `shift` + `p`
* Click on Install Package Control

<img width="600" alt="image" src="https://user-images.githubusercontent.com/3067731/192116256-61e0fe61-0c45-42da-8313-78947b0dc0d1.png">

Then you need to install packages with it
* `cmd` + `shift` + `p`
* Click on Package Control: Install Package

<img width="609" alt="image" src="https://user-images.githubusercontent.com/3067731/192116339-b32afeb9-d5b1-4172-ac20-b6cd02939674.png">


###### Package

Here are some packages that I like
* https://packagecontrol.io/packages/FileDiffs


#### Tabs

I don't like the way Sublime handles tabs, so I change it to make it more like Chrome. To do this, you'll need to edit your key bindings. Here's how you do that:

* (Windows or Linux) Preferences -> Key Bindings
* (Mac) Sublime Text -> Settings -> Key Bindings

Then add this to the file:
```
{ "keys": ["ctrl+tab"], "command": "next_view" },
{ "keys": ["ctrl+shift+tab"], "command": "prev_view" }
```
#### Markdown Files

I don't particularly like the way Sublime handles markdown files by default, so I change them. The exact menu location will depend on your operation system, but they're not too different.
* On a Mac, go into Preferences -> Settings -> Syntax Specific (Mac)
* On Linux, go into Preferences -> Settings - Syntax Specific

![png]({{site.baseurl}}/assets/img/sublime_linux_settings.png)

add this:
```
{
	"color_scheme": "Monokai.sublime-color-scheme",
	"draw_centered": false,
	"line_numbers": true,
	"gutter": true,
}
```
Here's another option:
```
{
	"color_scheme": "Packages/Monokai Extended/Monokai Extended.tmTheme",
	"dictionary": "Packages/Language - English/en_US.dic",
	"ignored_packages":
	[
		"Markdown",
		"Vintage"
	],
	"line_numbers": true,
	"show_full_path": true,
	"spell_check": true,
	"theme": "Adaptive.sublime-theme"
}
```
I always install these extensions:
* [Monokai Extended](https://github.com/jonschlinkert/sublime-monokai-extended)
* [Markdown Extended](https://github.com/jonschlinkert/sublime-markdown-extended)

For theme I do adaptive

Color scheme is monokai-extended

You'll want to make sure you used markdown-extended:
View -> Syntax -> Markdown Extended
but this only applies it to one file.

## Terminator

I customize Terminator to make it more like iTerm2. To customize it, you'll need to right click on the window then select "Preferences". Then go into Keybindings.
Here are some things I like to change:
* close_term
* new_window
* split_horiz
* split_vert

## VSCode

### Keybindings

There are several keybindings customizations I make in VSCode.

#### Make Tabs Work Like Chrome

I don't like the default behavior of tab cycling in VSCode because it switches tabs in order of most recently used, which I never remember. I prefer it to work like tabbing in Chrome, which is far more intuitive to me. Fortunately, VSCode lets you customize this. To change this, you'll need to edit your `keybindings.json` file like so:
* Open Command Palette (`cmd` + `shift` + `p` on a Mac)
* Search for "Preferences: Open Keyboard Shortcuts (JSON)"
* Open the file and add this to the file:

```
    {
        "key": "ctrl+tab",
        "command": "workbench.action.nextEditorInGroup"
    },
    {
        "key": "ctrl+shift+tab",
        "command": "workbench.action.previousEditorInGroup"
    },
```

![png]({{site.baseurl}}/assets/img/vscode_keyboard_shortcuts_windows.png)

The ones you're looking for are `workbench.action.nextEditor` and `workbench.action.previousEditor`.

#### Make .ipynb files work like Jupyter Notebooks

There's this annoying thing in VSCode where [you have to hit `esc` twice to exit a cell](https://github.com/microsoft/vscode/issues/121129) whereas you only have do to it once in Jupyter Notebooks. The solution is to add the following to your keybindings:
```
{
    "key": "escape",
    "command": "notebook.cell.quitEdit",
    "when": "notebookEditorFocused && !editorHasMultipleSelections && !editorHasSelection && !editorHoverVisible"
}
```

#### Accept Suggestion with Enter

```
    {
        "key": "enter",
        "command": "acceptSelectedSuggestion",
        "when": "suggestWidgetVisible && textInputFocus"
    },
    {
        "key": "tab",
        "command": "-acceptSelectedSuggestion",
        "when": "suggestWidgetVisible && textInputFocus"
    },
```

#### Final Keybindings

I have added other keybindings as well. My whole file looks like this:

```
[
    {
        "key": "ctrl+tab",
        "command": "workbench.action.nextEditorInGroup"
    },
    {
        "key": "ctrl+shift+tab",
        "command": "workbench.action.previousEditorInGroup"
    },
    {
        "key": "enter",
        "command": "acceptSelectedSuggestion",
        "when": "suggestWidgetVisible && textInputFocus"
    },
    {
        "key": "tab",
        "command": "-acceptSelectedSuggestion",
        "when": "suggestWidgetVisible && textInputFocus"
    },
    {
    "key": "escape",
    "command": "notebook.cell.quitEdit",
    "when": "notebookEditorFocused && !editorHasMultipleSelections && !editorHasSelection && !editorHoverVisible"
    }
]
```

### Syncing Settings Across Multiple Computers

I usually sync the following:

![image](https://user-images.githubusercontent.com/3067731/154163959-2cfa51c8-f760-46fd-b271-57db88ffd34c.png)

I usually sync with my Microsoft account, because I have different GitHub accounts linked for different computers.

### Snippets

I recommend using keyboard snippets. Here are some useful ones that I use:
```json
{
    "Import Matplotlib": {
        "prefix": [
            "plt"
        ],
        "body": [
            "from matplotlib import pyplot as plt"
        ],
        "description": "Import matplotlib"
    },
    "Save Image": {
        "prefix": [
            "save"
        ],
        "body": [
            "plt.imsave('$0.png', $0)"
        ],
        "description": "Save image"
    },
    "Show Image": {
        "prefix": [
            "show"
        ],
        "body": [
            "plt.imshow($0)",
            "plt.show()"
        ],
        "description": "Show image"
    },
    "Pytest": {
        "prefix": [
            "pytest"
        ],
        "body": [
            "if __name__ == '__main__':",
            "    pytest.main([__file__])"
        ],
        "description": "Set up Pytest"
    },
    "Import Polyplotter": {
        "prefix": [
            "ppp"
        ],
        "body": [
            "from polyplotter import plotpoly as p",
        ],
        "description": "Import the main function from polyplotter"
    }
}
```

### Settings

#### Make .ipynb files layout tigher
```
    // make notebook tighter
    "editor.lineHeight": 17,
    "notebook.insertToolbarLocation": "notebookToolbar",
    "notebook.showCellStatusBar": "hidden",
    "notebook.cellToolbarLocation": {
        "jupyter-notebook": "hidden"
    },
    "notebook.globalToolbar": false,
    "notebook.output.scrolling": true,
    "notebook.consolidatedRunButton": true,
```
## Cmder

You have to find your `%CMDER_ROOT%`. You can do this with `echo %CMDER_ROOT%`.


Go there and into config. It might be at `C:\Users\Julius\Downloads\cmder\config`




`"C:\Users\Julius\Downloads\cmder\config\user_aliases.cmd"`

It might start looking like:
```
;= @echo off
;= rem Call DOSKEY and use this file as the macrofile
;= %SystemRoot%\system32\doskey /listsize=1000 /macrofile=%0%
;= rem In batch mode, jump to the end of the file
;= goto:eof
;= Add aliases below here
e.=explorer .
gl=git log --oneline --all --graph --decorate  $*
ls=ls --show-control-chars -F --color $*
pwd=cd
clear=cls
history=cat "%CMDER_ROOT%\config\.history"
unalias=alias /d $1
vi=vim $*
cmderr=cd /d "%CMDER_ROOT%"
```

You can add your stuff to the bottom like so:

```
cdh=cd "C:\Users\Julius\Documents\GitHub"
```


## DataGrip

#### Execute Command

I find it's easier to use a hotkey that can easily be done with the left hand only, so I go with `command + R` to run queries. It also matches the default from Navicat, which is nice if you switch back and forth.

Go into Preferences -> Keymap and type in "execute" in the search bar. Then change `Execute (2)` from empty to `command + R`. This way you don't have to disable `shift + return`.

<img width="972" alt="image" src="https://user-images.githubusercontent.com/3067731/192343390-ecc29eb2-6982-48c8-be27-4ad2d7f0c087.png">

That hotkey is already assigned to "Refresh", but I remove it from there.

Note that you can also go into more detail with your `Execute` command in the Database -> Query Execution menu

<img width="974" alt="image" src="https://user-images.githubusercontent.com/3067731/192343012-b90c4e05-004f-4da3-ad83-d19b4d9b40f0.png">

#### Tabs

DataGrip also has a default tab process that I don't like. Here's how to update it to make it more like Chrome's:

* Go into Preferences -> Keymap and type in "switcher".

<img width="974" alt="image" src="https://user-images.githubusercontent.com/3067731/186535752-d7b07286-40e8-4512-921e-2994756d9b42.png">

* Double click on the keyboard shortcut icons on the right and remove them. Remove both of them.

* Type in "select tab" into the search and add the new commands to "Select Next Tab" and "Select Previous Tab" under Main Menu -> Window -> Editor Tabs

* Note that you can't just do the keyboard shortcut because it will think you are using `tab` to switch to the next box, so you have to click on the `+` on the right and select the appropriate hotkeys from the dropdown.

* It should look like this in the end:

<img width="984" alt="image" src="https://user-images.githubusercontent.com/3067731/186536462-34900e69-1f70-4344-b847-fd3bec14e2ac.png">

### Color-coding Databases and Folders

It can be really useful to color code your databases. This makes them automatically appear in the files menu in that color. To do so, you need to right click on a database or a subfolder in the Database Explorer (left-side panel). The follow the menu as you can see in the image below.

<img width="450" alt="image" src="https://user-images.githubusercontent.com/3067731/214940892-2d062527-7a8c-41e4-93e2-c302c8edf772.png">

You may need to restart DataGrip before the color changes take place in the Files menu (righthand side).

#### Refactor Code

I think the hotkey for refactoring code is not good, so I recommend double caps lock.

## DBeaver

I also don't like some of DBeaver's defaults.

![image](https://user-images.githubusercontent.com/3067731/197381400-ae4ec5e3-0df2-4777-928a-132a96dbcb1a.png)

You can change, for example, the "Execute SQL Statement" by going into Window -> Preferences -> User Interface -> Keys. Then you can change it to `Ctrl + R` (Windows) to match your other database tools.

## Karabiner Elements

Karabiner Elements is all about customization. I mainly use mine to make my experience across Mac, Linux, and Windows as seemless as possible.

* I recommend use the [Windows shortcuts on macOS](https://ke-complex-modifications.pqrs.org/?q=windows%20shortcuts%20on%20macOS). I use this to do a lot of thinks I like to do in Windows, like using `control` and the arrows to jump over words. Here's what it looks like:
<img width="1218" alt="image" src="https://user-images.githubusercontent.com/3067731/209397616-c8552408-665d-4745-b918-5e839dd1e91a.png">
* After you click "Import" you have to import it again within the app:
<img width="987" alt="image" src="https://user-images.githubusercontent.com/3067731/209397778-6ade395b-6140-4385-aece-6c55c563fea3.png">
* Then you have to enable it. If you scroll down to the bottom of that section, you'll find an "Enable All" button:
<img width="933" alt="image" src="https://user-images.githubusercontent.com/3067731/209398081-c04af843-f244-4b7f-a2bc-a27ea91219c0.png">
* I delete the shortcut that remaps `command` + `tab` because I like to keep that for switching between applications
<img width="779" alt="image" src="https://user-images.githubusercontent.com/3067731/209398188-5aa0bf22-a783-4912-9f8b-12ebb43b6f0f.png">

* The shortcuts will be stored under "Complex Modifications" in the app
