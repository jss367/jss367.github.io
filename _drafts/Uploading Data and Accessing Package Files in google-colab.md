# Uploading Data and Accessing Package Files in Google Colab



## 1. Uploading Your Own Data to Google Colab

Google Colab provides several ways to upload and access your data. Here's how you can upload a spreadsheet and open it with pandas:

### Method 1: Upload from Your Local Machine

1. In your Colab notebook, run the following code:

```python
from google.colab import files
uploaded = files.upload()
```

2. A file picker will appear. Select your spreadsheet file (e.g., 'data.csv').

3. After the upload is complete, you can read the file using pandas:

```python
import pandas as pd
import io

df = pd.read_csv(io.BytesIO(uploaded['data.csv']))
print(df.head())
```

### Method 2: Access from Google Drive

1. Mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Follow the authentication steps.

3. Once mounted, access your file:

```python
import pandas as pd

file_path = '/content/drive/My Drive/path/to/your/file.csv'
df = pd.read_csv(file_path)
print(df.head())
```

## 2. Accessing Files Within an Installed Package

To access files within an installed package, you can use Python's `importlib` module. Here's how you can access a file like 'autotuneml/configs/run_config.yaml':

1. First, ensure the package is installed:

```python
!pip install autotuneml
```

2. Use `importlib.resources` to access the file:

```python
from importlib import resources
import autotuneml

# Read the content of the file
with resources.path('autotuneml.configs', 'run_config.yaml') as config_path:
    with open(config_path, 'r') as config_file:
        config_content = config_file.read()

print(config_content)
```

3. If you need to work with the YAML content, you can use the `yaml` library:

```python
import yaml
from importlib import resources
import autotuneml

with resources.path('autotuneml.configs', 'run_config.yaml') as config_path:
    with open(config_path, 'r') as config_file:
        config_data = yaml.safe_load(config_file)

print(config_data)
```

This method works for any installed package, not just autotuneml. Replace 'autotuneml' and the file path with the appropriate package and file you're trying to access.

Remember that the exact file structure may vary depending on the package, so you might need to adjust the path accordingly.