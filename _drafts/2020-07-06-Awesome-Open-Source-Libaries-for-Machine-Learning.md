---
layout: post
title: "Python Cheatsheet II"
description: "Awesome Open Source Libaries for Machine Learning"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/docksunrise.jpg"
tags: [Python, Cheatsheet]
---


# Frameworks

You've got to start with frameworks, and the deep learning world

* Tensorflow - Google's tool for deep learning. The dominate tool in the industry.

* PyTorch - Facebook's deep learning framework. Incredibly powerful and great for research.

* [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - High-level interface for PyTorch. It's a bit like "Keras for PyTorch".

* FastAI - Built on top of PyTorch. I've used the first version and there's a lot of great stuff in it. Yet to dig into the second version.

# Sharing Models

* [ONNX](https://onnx.ai/) - Designed to make sharing models easier. I'll add a quick tutorial at some point and update this.


https://www.tensorflow.org/tfx

# Deployment

* [Jenkins](https://www.jenkins.io/) - The go-to for automating your deployment pipeline. Great for continuoius integration / continuous delivery. Jenkins can do so much 

https://github.com/snorkel-team/snorkel
argo: https://argoproj.github.io/
kubernetes



ONNX
https://github.com/dessa-oss/atlas


bentoml:
mlflow
* Note: despite MLFlow being decently well established, I still found it a little buggy. For example, which I did `conda install mlflow` on my Windows machine and tried to use it I got an error. I had to 
```
conda install mlflow --only-deps
pip install mlflow
```
per [this issue](https://github.com/mlflow/mlflow/issues/1951) for it to work.



kubeflow
https://deepnote.com/
https://github.com/google/jax

https://github.com/Netflix/metaflow

### Auto Hyperparameter Tuning

[Katib](https://github.com/kubeflow/katib) - Built specifically on top of kubeflow


### Privacy

[PySyft](https://github.com/OpenMined/PySyft) - I haven't looked at this yet but looks interesting


koalasThe Koalas project makes data scientists more productive when interacting with big data, by implementing the pandas DataFrame API on top of Apache Spark.
tensorboard

API Generators:

* [Swagger](https://swagger.io/)

* [Open API Generator](https://github.com/OpenAPITools/openapi-generator)
