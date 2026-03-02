---
layout: post
title: "Linux Setup"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/turtle.jpg"
tags: [Linux, Software]
last_modified: 2026-02-27
---

This post covers how I set up my shell and development environment on Linux. I use Windows, Mac, and Linux regularly, so I try to keep things consistent across platforms where I can. You can see the [software I use](https://jss367.github.io/software-i-use.html) and [how I customize it](https://jss367.github.io/software-customizations.html) in the linked posts; this one focuses on shell and terminal setup.

<b>Table of Contents</b>
* TOC
{:toc}

## Shell

On Linux, I use [zsh](https://www.zsh.org/) as my shell. Unlike macOS, zsh isn't the default on most Linux distributions so you'll need to install it and set it as your default shell:

```bash
sudo apt install zsh
chsh -s $(which zsh)
```

Log out and back in for the change to take effect. You can verify with `echo $SHELL`.

My setup is built on top of [Oh My Zsh](https://ohmyz.sh/), which provides a nice framework for managing zsh configuration, themes, and plugins. You can check if it's installed with `ls ~/.oh-my-zsh`.

## Development Environment

I store my configuration in a [dotfiles repo](https://github.com/jss367/dotfiles) so I can sync it across machines. The core idea is simple: put all your portable configuration (aliases, functions, environment variables) in a single `~/.profile` file and symlink it from the repo. Then source it from whatever shell you're using.

### The Sourcing Chain

Here's how my config loads:

```
~/.zshrc  ->  oh-my-zsh auto-sources ~/.oh-my-zsh/custom/profile.zsh  ->  ~/.profile  ->  ~/.credentials
```

The pieces:

1. **`~/.zshrc`** loads Oh My Zsh, which automatically sources every `.zsh` file in `~/.oh-my-zsh/custom/`.
2. **`~/.oh-my-zsh/custom/profile.zsh`** is a one-line file: `source ~/.profile`. This is the bridge between zsh and your portable config. You need to create this file manually on each new machine.
3. **`~/.profile`** is where all the real configuration lives — aliases, functions, PATH modifications. This is the file I symlink from my dotfiles repo.
4. **`~/.credentials`** contains passwords and API keys. It's sourced from `.profile` but never committed to any repo.

### Don't Symlink .zshrc

I don't recommend making your `.zshrc` portable. Many tools modify `.zshrc` as part of their installation:

* `conda init` adds a conda initialization block
* nvm adds its loader
* Various tools append PATH entries

Each of these will either overwrite your symlink with a regular file or modify your repo's copy with machine-specific paths. It's a constant battle.

Instead, I leave `.zshrc` as a regular local file on each machine. The things you actually want to sync — your aliases, functions, and shortcuts — belong in `.profile`. Let `.zshrc` handle the machine-specific stuff (Oh My Zsh setup, conda init, tool-specific PATH additions).

### Setting Up a New Machine

```bash
# Clone dotfiles
git clone https://github.com/YOUR_USER/dotfiles.git ~/git/dotfiles

# Symlink .profile
ln -sf ~/git/dotfiles/shell/profile ~/.profile

# Symlink .gitconfig
ln -sf ~/git/dotfiles/git/gitconfig ~/.gitconfig

# Create the oh-my-zsh bridge (one-time setup)
echo 'source ~/.profile' > ~/.oh-my-zsh/custom/profile.zsh
```

Then open a new terminal or run `source ~/.zshrc`.

## Terminal

Linux has many terminal emulators to choose from. The default terminal that comes with your desktop environment (GNOME Terminal, Konsole, etc.) works fine.

## Packages

Here are the packages I use to improve my terminal experience.

### powerlevel10k

[powerlevel10k](https://github.com/romkatv/powerlevel10k) is a fast, customizable zsh theme. Install it through Oh My Zsh (not through apt), then start a new terminal and it will walk you through configuration.

### zoxide

[zoxide](https://github.com/ajeetdsouza/zoxide) is a smarter `cd` that learns your most-used directories. After visiting a directory once, you can jump to it by typing any part of its name. Install it with:

```bash
sudo apt install zoxide
```

I alias it to `j`:

```bash
alias j='z'
alias ji='zi'  # interactive mode with fzf
```

Add to your `.zshrc`:

```bash
eval "$(zoxide init zsh)"
```

### fzf

[fzf](https://github.com/junegunn/fzf) is a general-purpose fuzzy finder. It enhances ctrl+r history search, file finding, and more. Add to your `.zshrc`:

```bash
source <(fzf --zsh)
```

### zsh-autosuggestions

[zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions) suggests commands as you type based on your history. Install it, then add it to your Oh My Zsh plugins in `.zshrc`:

```bash
plugins=(git zsh-autosuggestions)
```

### zsh-syntax-highlighting

[zsh-syntax-highlighting](https://github.com/zsh-users/zsh-syntax-highlighting) highlights valid commands as you type — green for valid, red for invalid. Install via apt:

```bash
sudo apt install zsh-syntax-highlighting
```

Then source it in your `.profile`:

```bash
source /usr/share/zsh-syntax-highlighting/zsh-syntax-highlighting.zsh
```

### Pygments

[Pygments](https://pygments.org/) is a Python syntax highlighter. It's like `cat` with colors. I alias it to `ccat`:

```bash
alias ccat='pygmentize -O style=monokai -f console256 -g'
```

## Keyboard Customizations

In zsh, you may need to configure word-jumping with Alt+Arrow. Add this to your `.profile`:

```bash
bindkey "\e\e[D" backward-word
bindkey "\e\e[C" forward-word
```

The exact key codes depend on your terminal emulator — the above works in most common setups. If it doesn't, use `cat -v` and press the key combination to see the escape sequence your terminal sends.

## Conda

I typically install Miniconda for a single user, which places it at `~/miniconda3/`. When you run `conda init zsh`, it adds an initialization block to your `.zshrc`:

```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/$USER/miniconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/$USER/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/$USER/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/$USER/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```

Leave it there.

### Conda and tmux

If you use `tmux`, be aware that it doesn't always source `.zshrc` — sometimes it only sources `.profile`. This means conda won't load in tmux sessions, and you might end up using the system Python instead. Check with `which python`. If this happens, move the conda initialization block from `.zshrc` into `.profile`.

## Working on Remote Linux Instances

If you're SSHing into a remote Linux machine, you'll likely be working with bash rather than zsh. You'll have a `.bashrc` that contains your prompt, conda init, and other configuration.

You'll also want to put your `.credentials` file there and source it from `.bashrc`. If you need to distinguish paths between local and remote machines, you can add something like `export REMOTE_BASE="/home/me"` to your credentials file.

### Credentials on Remote Machines

You might want environment variables in your credentials file that adapt to the environment:

```bash
export LOCAL_PATH="/home/me"
export REMOTE_PATH="/home/me-on-remote"

if [ -n "$ZSH_VERSION" ]; then
    export TRUE_PATH=$LOCAL_PATH
elif [ -n "$BASH_VERSION" ]; then
    alias rld='source ~/.bashrc'  # reload profile
    export TRUE_PATH=$REMOTE_PATH
else
    echo "Warning: Shell unknown"
fi
```

## Bash

I still use bash fairly often, especially on remote servers. Because bash doesn't come with the same built-in aliases and features that Oh My Zsh provides, I add some extras manually.

### History and Key Bindings

```bash
if [ "$SHELL" = "/bin/bash" ]; then
    shopt -s histappend                      # append to history, don't overwrite it
    bind '"\e[A":history-search-backward'
    bind '"\e[B":history-search-forward'
fi

export HISTSIZE=1000000
export HISTFILESIZE=1000000000
export HISTCONTROL=ignoredups:erasedups      # no duplicate entries
```

### Navigation Aliases

```bash
alias ..='cd ..'
alias ...='cd ../../../'
alias ....='cd ../../../../'
```

### Git Prompt

I like to customize the bash prompt to show the current git branch:

```bash
parse_git_branch() {
    git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}
export PS1="\u@\h \[\e[32m\]\w \[\e[91m\]\$(parse_git_branch)\[\e[00m\]$ "
```

### Color Support

```bash
if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
    alias fgrep='fgrep --color=auto'
    alias egrep='egrep --color=auto'
fi
```

## Alias Notes

When using shell variables in aliases, quoting matters. If you define a variable:

```bash
export BASE="$HOME/git"
```

You need double quotes in the alias so the variable gets expanded:

```bash
alias cdh="cd $BASE"  # works — $BASE is expanded
alias cdh='cd $BASE'  # won't work — $BASE stays literal
```

However, `$HOME` works with single quotes in most shells because it's expanded at a different stage.

## Debugging Your Environment

If something unexpected shows up in your environment, you can search for it across your shell config files:

```bash
grep 'mysterious_string' ~/.zshrc ~/.zshenv ~/.profile ~/.bashrc ~/.bash_profile
```

## Useful Git Commands

A compact, colorized git log format:

```bash
git log --pretty=format:'%C(yellow)%h %Cred%ad %Cblue%an%Cgreen%d %Creset%s' --date=short
```

## Full .profile

My full `.profile` is available in [my dotfiles repo](https://github.com/jss367/dotfiles/blob/main/shell/profile).
