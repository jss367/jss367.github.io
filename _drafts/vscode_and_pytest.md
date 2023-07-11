
VSCode and Pytest aren't perfect together. Sometimes I won't be able to run all my tests in VSCode, despite being able to run them from the command line. Here are some tips for working through this issue.

If your path is 
```
        {
            "path": "my_monorepo"
        },
```

Then your test cases will run like `my_project/tests/unit/test_helpers.py` and they will work.

But if it's a monorepo and you want your paths to look like:

```
{
    "folders": [
        {
            "path": "my_monorepo/my_project_a"
        },
        {
            "path": "my_monorepo/my_project_b"
        },
    ],
}
```


Then your test cases will run like `my_project_a/tests/unit/test_helpers.py` and it might fail. That's because it has `my_project_a` twice in the path.

The reason for this seems to be because of a distinction VSCode makes between the workspace root and the project root directories (I think).

A way to handle this scenario would be to create a `.env` file in each of your projects to set the `PYTHONPATH` environment variable. This way you can specify exactly where your Python files are and pytest should use that for its base path.

Here is how you can do it:

In each of your project folders (my_project_a and my_project_b), create a .env and a pytest.ini file.

```env
PYTHONPATH=${workspaceFolder}
```

```
[pytest]
testpaths =
    tests
```

The weird thing I've found is once you create those, you can comment them out. So it's like you just need to clean something up or something.
