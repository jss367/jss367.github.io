mypy is a great static type checker, but I'm found to get the best value out of it you need to customize it a bit. The main issue I have with it is it prints out too much, much of which is unnecessary.

For example, you'll often see this message:
```
error: Skipping analyzing "pandas": module is installed, but missing library stubs or py.typed marker  [import]
```


One solution is to list the packages to ignore in a config file. This isn't a great solution because then you need to lug config files with `pandas`, `numpy`, etc. around everywhere with you.

You can do `mypy . --ignore-missing-imports` but this isn't a great solution because it ignores too much. See the conversation here: https://github.com/python/mypy/issues/9789


If you're trying to run mypy on a package that has hyphens in the name, you'll have to temporarily change those to underscores.
