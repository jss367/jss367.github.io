---
layout: post
title: "Path Problems: Mac"
description: "A guide to some of the path problems you may face on Macs"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/dark_path.jpg"
tags: [Mac, Python]
---

Path problems are some of the most common and annoying problems machine learning engineers face, especially when frequently switching between operating systems. There are so many different issues ways to have path problems that no post could cover them all, but in this post, I’ll try to provide some background on possible issues and how to resolve them. Whether you're running scripts, installing new software, or managing Python projects, understanding how these paths work will save you countless hours of troubleshooting and configuration.

The first thing you need to realize are that there are multiple different things that are at times called a "path". The two main ones that Python programmers will run into are the system path and the Python path. System paths, represented by the `$PATH` environment variable, determine where your shell looks for executable files when you type a command in the terminal. On the other hand, Python paths, determined by `sys.path` and influenced by the `$PYTHONPATH` environment variable, specify the locations where Python searches for modules and packages during the import process. `$PATH` is for executable binaries; `$PYTHONPATH` is for Python modules.

<b>Table of Contents</b>
* TOC
{:toc}

## Understanding System Paths ($PATH)

### What is `$PATH`?

The `$PATH` environment variable is a fundamental concept in Unix-based operating systems that tells the shell where to look for executable files. It is a colon-separated list of directories that your shell searches when you enter a command in the terminal. When you type a command, the shell looks for an executable file with that name in the directories listed in your `$PATH`, in the order they appear.

### How the system uses `$PATH` to locate executables

When you run a command like `python` or `git`, the shell searches through the directories specified in your `$PATH` to find the corresponding executable file. It checks each directory in the order they are listed until it finds a match. If the executable is found, it is executed; otherwise, you'll see an error message indicating that the command was not found.

### Viewing your current $PATH

To view your current `$PATH`, you can use the `echo` command followed by `$PATH` in the terminal:

```bash
echo $PATH
```

The output will be a colon-separated list of directories, for example:

```bash
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```

It might be a little hard to read, so if you want a more readable version you can use:

```bash
echo "${PATH//:/$'\n'}"
```

On a brand new Mac your path is:

```
julius@Juliuss-MacBook-Pro ~ % echo $PATH
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```

Each directory in this list is searched in order when you enter a command.

### Adding directories to $PATH

If you have executables in a directory that is not listed in your `$PATH`, you can add it using the export command:

```bash
export PATH=$PATH:/path/to/directory
```

This command appends the specified directory to the end of your current $PATH. However, this change is temporary and will be lost when you close the terminal session.

To persist the change across sessions, you can add the export command to your shell configuration file (e.g., .bashrc or .zshrc, but [I recommend `.profile`](https://jss367.github.io/shell-and-environment-setup.html)). Open the file in a text editor and add the export command at the end. Save the file and restart your terminal or run source ~/.bashrc (or source ~/.zshrc) to apply the changes.

Note:
If you change the environment variables you'll need to restart VSCode or Jupyter Notebooks. Once you restart you'll see the addition in `sys.path`

## Python Path (sys.path and $PYTHONPATH)

### What is sys.path?

`sys.path` is a list of directory paths that Python uses to search for modules when you use the `import` statement. If you can't import a module you wrote, it's probably because that location is missing from your path. When you import a module, Python looks for the module in the directories listed in `sys.path`, in the order they appear.

#### Python's search path for modules

When you run a Python script, the directory containing the script is automatically added to the beginning of `sys.path`. Python then searches for modules in the following locations, in this order:

1. The directory containing the script being run
2. The directories listed in the `$PYTHONPATH` environment variable (if set)
3. Standard library directories
4. Directories added during installation or by third-party packages

### Viewing sys.path

To view the current `sys.path`, you can run the following command in the terminal:

```bash
python -c "import sys; print(sys.path)"
```

This command starts a Python interpreter, imports the `sys` module, and prints the contents of `sys.path`.
The output will be a list of directory paths, for example:


```
['', '/usr/local/lib/python3.9/site-packages', '/usr/local/lib/python3.9', ...]
```


### $PYTHONPATH environment variable

1. How `$PYTHONPATH` affects `sys.path`

The `$PYTHONPATH` environment variable is a colon-separated list of directories that Python adds to `sys.path` when the interpreter starts. By setting `$PYTHONPATH`, you can add custom directories to Python's search path, making it easier to import modules that are not in the standard library or installed in the default locations.

2. Setting $PYTHONPATH to add custom module directories

To set $PYTHONPATH, you can use the export command in the terminal:

```bash
export PYTHONPATH=/path/to/directory1:/path/to/directory2
```

This command sets `$PYTHONPATH` to a colon-separated list of directories. You can add multiple directories by separating them with colons.
Like with $PATH, you can make this change persistent by adding the export command to your shell configuration file (e.g., .bashrc or .zshrc).

### Adding directories to sys.path programmatically

#### Using sys.path.append('/path/to/directory')

You can also add directories to `sys.path` programmatically within your Python script using the `sys.path.append()` method:

```python
import sys
sys.path.append('/path/to/directory')
```

This approach is useful when you need to add a directory to the search path temporarily or conditionally based on your script's logic.

#### Importance of modifying sys.path carefully

When modifying `sys.path`, it's crucial to be cautious and consider the potential consequences. Adding directories to the beginning of `sys.path` can shadow modules from the standard library or other installed packages, leading to unexpected behavior. It's generally safer to append directories to the end of `sys.path` to avoid such issues.


### ModuleNotFoundError

#### Explanation of the error

`ModuleNotFoundError` is a common error that occurs when Python cannot find a module you are trying to import. This error indicates that the module is not present in any of the directories listed in `sys.path`.

#### Common causes related to Python path issues

Some common causes of `ModuleNotFoundError` related to Python path issues include:

* The module is not installed in any of the directories listed in `sys.path`
* The module is installed in a directory that is not listed in `sys.path`
* The module name is misspelled or does not match the actual module filename
* The `$PYTHONPATH` environment variable is set incorrectly, causing Python to search in the wrong directories

To resolve `ModuleNotFoundError`, you should ensure that the module is installed correctly and that the directory containing the module is included in `sys.path` or `$PYTHONPATH`.


## Identifying Python Executable (sys.executable)

### What is sys.executable?

`sys.executable` is a string variable in the Python `sys` module that contains the absolute path to the Python interpreter currently running your script. It is useful when you need to know the exact location of the Python executable being used, especially when working with virtual environments or multiple Python versions.

### Using sys.executable to locate the current Python interpreter

To print the path to the current Python interpreter, you can use the following code:

```python
import sys
print(sys.executable)
```


When you run this script, it will output the absolute path to the Python executable, for example:

```
/usr/local/bin/python3
```

This information can be helpful when you need to:

* Verify that you are running the correct version of Python
* Set up a virtual environment using the same Python interpreter
* Run a Python script using a specific interpreter
* Troubleshoot issues related to multiple Python installations

### Ensuring the correct Python version is used

When working on projects with specific Python version requirements, it's crucial to ensure that you are using the correct Python interpreter. By checking sys.executable, you can confirm that your script is running under the expected Python version.

If you find that the Python interpreter being used is not the one you intended, you can take the following steps:

* Check your `$PATH` to ensure that the desired Python version's directory is listed and appears before other Python installations.
* Use a virtual environment to create an isolated Python environment with the required version.
* Explicitly specify the Python interpreter when running your script, for example:
```bash
/path/to/desired/python script.py
```

By being aware of `sys.executable` and taking steps to ensure the correct Python version is used, you can avoid compatibility issues and ensure that your scripts run as expected.


## Best Practices and Tips

### Keeping system path and Python path organized

To maintain a clean and organized development environment, it's essential to keep your system path (`$PATH`) and Python path (`sys.path` and `$PYTHONPATH`) well-structured. Here are some tips:

- Only add directories to `$PATH` that contain executables you need to access globally.
- Use a consistent naming convention for custom module directories added to `$PYTHONPATH`.
- Document any custom path modifications in your project's README or documentation.

### Avoiding excessive modification of system-wide paths

While it's sometimes necessary to modify system-wide paths like `$PATH` or `$PYTHONPATH`, it's best to do so sparingly. Excessive modifications can lead to unexpected behavior, conflicts between packages, and difficulties in reproducing your development environment on other machines.

Instead of modifying system-wide paths, consider:

* Using virtual environments for project-specific dependencies and paths
* Installing packages in user-specific directories (e.g., `$HOME/.local/bin`) instead of system-wide locations
* Using package managers like `pip` with the `--user` flag to install packages in user-specific directories

### Troubleshooting path-related issues

If you encounter issues related to system or Python paths, here are some troubleshooting steps:

* Double-check your `$PATH` and `$PYTHONPATH` to ensure they contain the expected directories in the correct order.
* Verify that the required packages are installed in the correct locations and accessible from your Python environment.
* Check for conflicts between package versions or identically named modules in different directories.
* Use `sys.path` and `sys.executable` to confirm that your script is running in the intended Python environment.
* Consult the documentation or seek help from the community if you're unsure about how to resolve a specific path-related issue.

By following these best practices and tips, you can keep your development environment organized, avoid common path-related issues, and ensure that your Python projects run smoothly.

## Recap

- The system path (`$PATH`) is a list of directories where the shell looks for executables when you enter a command in the terminal.
- Python paths (`sys.path` and `$PYTHONPATH`) determine where Python searches for modules when you use the `import` statement.
- You can view and modify your `$PATH` and `$PYTHONPATH` using commands like `echo` and `export`.
- `sys.executable` provides the absolute path to the current Python interpreter, helping you ensure that you're using the correct Python version.



