---
layout: post
title: "Software Customizations"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/car.jpg"
tags: [Linux, Mac, Software, Windows]
---

This post details some of the customizations I have made to some of the [software I use](https://jss367.github.io/software-i-use.html) in that hopes that it is helpful to others.

<b>Table of Contents</b>
* TOC
{:toc}

## Sublime Text

#### External Packages

Sublime Text has a great ecosystem of packages and plugins. Here's how to open the package manager:
* `cmd` + `shift` + `p`

###### Package

Here are some packages that I like
* https://packagecontrol.io/packages/FileDiffs



#### Tabs

I don't like the way Sublime handles tabs, so I change it to make it more like Chrome. To do this, you'll need to edit your key bindings. Here's how you do that:

* (Windows or Linux) Preferences -> Key Bindings
* (Mac) Sublime Text -> Preferences -> Key Bindings

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

I did the same thing with tabs in VSCode.

### Make Tabs Work Like Chrome

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
]
```

![png]({{site.baseurl}}/assets/img/vscode_keyboard_shortcuts_windows.png)

The ones you're look for are `workbench.action.nextEditor` and `workbench.action.previousEditor`.

### Syncing Settings Across Multiple Computers

I usually sync the following:

![image](https://user-images.githubusercontent.com/3067731/154163959-2cfa51c8-f760-46fd-b271-57db88ffd34c.png)

I usually sync with my microsoft account, because I have different github accounts linked for different computers.



## Cmder

You have to find your `%CMDER_ROOT%`. You can do this with `echo %CMDER_ROOT%`.


Go there and into config. It might be at `C:\Users\Julius\Downloads\cmder\config`




"C:\Users\Julius\Downloads\cmder\config\user_aliases.cmd"

It might start be looking like:
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

DataGrip also has a default tab process that I don't like. Here's how to update it to make it more like Chrome's:

* Go into Preferences -> Keymap and type in switcher.

<img width="974" alt="image" src="https://user-images.githubusercontent.com/3067731/186535752-d7b07286-40e8-4512-921e-2994756d9b42.png">

* Double click on the actions to remove them. Remove both of them.

* Type in "select tab" into the search and add the new commands to "Select Next Tab" and "Select Previous Tab" under Main Menu -> Window -> Editor Tabs

* It should look like this in the end:

<img width="984" alt="image" src="https://user-images.githubusercontent.com/3067731/186536462-34900e69-1f70-4344-b847-fd3bec14e2ac.png">

