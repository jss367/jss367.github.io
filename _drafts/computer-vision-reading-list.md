All the papers to read:


## The classics

The Viola-Jones papers
[Rapid Object Detection using a Boosted Cascade of Simple Features](https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/viola-cvpr-01.pdf)

The HOG paper
The DPM (deformable parts model) paper
Papers specifically on computer vision for remote sensing:


## Deep Learning Revolution
AlexNet - first use of ReLU
- used norm layers - not common anymore
-- might be a good example for pros and cons of reading papers
-- should also mention that you can find blogs that link to papers on arxiv now



VGG - stacking building blocks of the same shape
VGG: Leveraging repeating layers to build a deep architecture model
* The VGG paper
** Learn about how to go deeper through batch norm


ResNet: Introducing a shortcut from the previous layer to the next layer

Inception: Following split-transform-merge practise to split the input to multiple blocks and merging blocks later on.

ResNeXt: The principle is stacking the same topology blocks. Within the residual block, hyper-parameters (width and filter sizes) are shared.



Deeper networks are doing better, but they run into problems, like the vanishing gradient problem:
he vanishing gradient
problem (Zagoruyko & Komodakis, 2016

to solve that we have
 skip connections (He et al., 2016
 )
and batch normalization (Ioffe & Szegedy, 2015




Broad overview of deep learning:

Deep Learning Book (Bengio & Goodfellow)

"Practical Recommendations ..." paper from Bengio (a little bit outdated, but most of it holds up, many useful "tricks" of the trade)
Gradient descent techniques


* The Googlenet/Inception paper and the v4 follow up (skipping over v2 and v3)
** ResNet
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

* Fully convolutional nets
* U-net paper
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) - 18 May 2015

Dilated convolutions

The ResNet paper and ResNet analysis paper
"How transferable are features in deep neural networks?" (empirical studies of fine-tuning)

The dropout paper
The FaceNet paper (triplet loss / learning an embedding)
Faster-RCNN (region proposal + detection with pure CNNs)
SSD paper and talk
Google's paper comparing different object detection methods
DeepMask (instance segmentation)
Batch normalization
[YOLO paper: You only look once: Unified, real-time object detection](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Redmon_You_Only_Look_CVPR_2016_paper.pdf) (2016), J. Redmon et al
There were improvements to it. Specifically, YOLOv3 is worth reading:
[YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
Spatial pyramid pooling in deep convolutional networks for visual recognition (2014), K. He et al.
ML systems/infra papers/blog posts:

Google's TFX
Google's ML technical debt paper
Uber's ML infra
Facebook's ML infra
CV at pinterest
The classics (not directly relevant, but still worth reading, again for historical context):


Deep Learning in Remote Sensing: A Review (2017)
Towards Better Exploiting Convolutional Neural Networks for Remote Sensing Scene Classification (2016)
Not so essential, but still worth reading:

Unsupervised Visual Representation Learning by Context Prediction ( Self-supervised learning) (ICCV 2015)
End-to-End Instance Segmentation with Recurrent Attention (2017)

Other Computer Vision courses at Stanford with relevant materials:

CS331B: Representation Learning in Computer Vision: Research paper reading class, follow-up to CS231N. links to a good selection of CV papers
CS231A: Computer Vision, From 3D Reconstruction to Recognition
There is also a crowdsourced GitHub repo keeping track of the "top-100" papers in deep learning: https://github.com/terryum/awesome-deep-learning-papers


## Topics

#### Initialization:
Xavier:
 http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

He:
https://arxiv.org/pdf/1502.01852.pdf




the cv roadmap

feature pyrmamid networks



which extracts
feature maps through a feature pyramid, thus facilitating object detection in different scales, at a marginal extra cost

Adam

 hard-negative mining together with concatenated and deconvolutional feature maps.


 SSD

 SSD with FPN:  hard-negative mining together with concatenated and deconvolutional feature maps.


 hard negative mining:
 Training Region-based Object Detectors with Online Hard Example Mining






CV road map
I don't know if it's the best idea, but at some point you may want to. Once you've gotten fairly deep I think it's a necessity. Some reasons it may not be the best idea :
Fairly popular ideas are often better summarized in blogs
Some important papers are poorly written
They might have more ideas in the paper than the important one
Dropout 
Batch norm
Momentum ILYa sursekever 2013
Transfer learning
Resnet
Inception
Alexnet
Added relu, Dropout
Basic structure of convnets
Fpn
1 X 1 convolution
Commonly used to change the depth of input without affecting Spatial structure
Vgg
More small convolutionals to get the same effective receptive field with fewer parameters. Also has more relu which means more non linear operations
Relu
Scale jittering
New ways to deal with vanishing gradients?
Googlenet
Inception blocks
- network in network
 Resnet
-added path for identity mapping
- skip path and residual path
- mostly ways to go deeper without losing network performance
Viola-Jones
-integral image
- “learned” features selection - used adaboost to find most helpful features
- detection cascades - pass from one stage to another; spend less time on obvious “not-faces”
HOG Detector
Deformable Part-based Model (DPM)
-mixture models, hard negative mining, bounding box regression
- these are claimed to be CNNs: https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Girshick_Deformable_Part_Models_2015_CVPR_paper.pdf
Overfeat
a pioneer model of integrating the object detection, localization and classification tasks all into one convolutional neural network
Architecture similar to AlexNet
Regression network to output bbox locations
RCNN
Generally start with a pre-trained Alexnet
-selective search, rescaled, fed into CNN (e.g. alexnet); svm to classify result
- selective search proposes category-independent regions
- but you have to make them fixed size to feed into alexnet (because of the dense layers are the end), so you have to warp them to a fixed size
- continue fine-tuning the CNN on warped proposal regions for K + 1 classes; The additional one class refers to the background (use smaller learning rate, oversample positive classes to avoid too much background)
- then binary SVM trained independently for each class

SPPNet
Adds Spatial Pyramid Pooling (SPP) layer - enables a CNN to generate a fixed-length representation regardless of the input
Computes feature maps once and then repeatedly reuses
Fast RCNN
Allows training to be much easier than sppnet
Faster RCNN
Region Proposal Network
Some speedups to Faster RCNN: RFCN and Light head RCNN
FPN
Based on Faster RCNN
Describe what it does!
This is a basic building block (or architecture) of many of today’s models
YOLO, 2,3
Extremely fast; fastest version can run at 155 fps (most video is 30 or 60 fps)
SSD
introduction of the multi-reference and multi-resolution detection techniques
Detects different scales on different layers of the network
RetinaNet
Deals with foreground-background class imbalance
Papers associated with datasets. I don’t think these are must-reads but the datasets are good to be familiar with. 
Pascal VOC (VOC2007, VOC2012)
ImageNet
MSCOCO



Going deeper with convolutions
Rethinking the Inception Architecture for Computer Vision

lenet-5 - 1998 Gradient-based learning applied to document recognition

dannet

Identity Mappings in Deep Residual Networks
- better architecutre for resnet



This site also has a lot: https://github.com/floodsung/Deep-Learning-Papers-Reading-Roadmap

