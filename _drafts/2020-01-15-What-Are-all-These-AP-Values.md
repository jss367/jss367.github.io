---
layout: post
title: "What Are all These Different AP Values"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/palm_sunset.jpg"
tags: [Python]
---

In recent years, the most frequently used evaluation for object detection is “Average Precision (AP)”, which was originally introduced in [The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) (VOC2007). It's a relatively simple concept, but when you look at research papers, you'll often see lots of variations on the basic concept. This post will clarify all those variations.

First, let's be clear about average precision. The term is exactly what it sounds - the average precision of a model. If you check the precision at a bunch of different thresholds, it's the average of them. So, how do you calculate it? You have to change the threshold to keep increasing the recall, then find the precision at each point. To make the curve decrease monotonically, you just Note that if you actually do this precision doesn't decrease monotonically. That's OK. You make the curve monotonic by setting the precision to be the highest value at that given recall or higher. This is because you could always tweak the threshold to a value that's going to give you higher precision and recall.

OK, so you know that "Average Precision", or AP, is the area under the precision recall curve. But when you look at a research paper, such as Cascade Mask R-CNN (reproduced below), they have a bunch more than than.


What does map@(0.5:0.95)mean? 
Page 157 of tf book
Example: retinanet: COCO mAP@.5=59.1%, mAP@[.5, .95]=39.1%


## Multiclass classification

Many datasets have lots of objects. [COCO](http://cocodataset.org/#home), for example, has 80. 

To compare performance over all object categories, the mean AP (mAP) averaged over all object categories is usually used. 
AP is area under curve for a single class. When multiclass, it’s combined into mAP
mAP is between 0 and 100, with higher numbers better

AP is defined as the average detection precision under different recalls, and is usually evaluated in a category specific manner




To measure the object localization accuracy, the Intersection over Union (IoU) is used to check whether the IoU between the predicted box and the ground truth box is greater than a predefined threshold, say, 0.5. If yes, the object will be identified as “successfully detected”, otherwise will be identified as “missed”. 

The 0.5- IoU based mAP has then become the de facto metric for object detection problems for years.

But these are really different things. The threshold you use will influence which types of models appear to perform best. 0.5 threshold is very coarse localization while 0.95 is very precise.

## COCO AP
Instead of using a fixed IoU threshold, MS-COCO AP is averaged over multiple IoU thresholds between 0.5 (coarse localization) and 0.95 (perfect localization). This change of the metric has encouraged more accurate object localization and may be of great importance for some real-world applications

## Other attempts

There are other approaches but they haven’t taken off: https://arxiv.org/pdf/1807.01696.pdf


What is Box AP? - I think it’s the same as AP. I guess it just means with bounding boxes. As opposed to with Masks
“The proposed methods are only applied on the detection branch in Mask R-CNN. APbb means the detection performance and APmask indicates the segmentation performance”
There’s also APs, APm, APl - small, medium, large
Maybe look at example from cascade mask rcnn or something


AP, AP50, AP75, mAP, AP@[0.5:0.95],

AP75 is simply the AP with a threshold of 75.

In COCO they change the IoU values from 50% to 95%, at a step of 5%.
So you get 10 different values.. The mean of those values is the AP@[0.5:0.95].

There is not perfect consistency here. There is very good consistency, especially when benchmarking on datasets like COCO. 


Sometimes you'll see APbbox or BoxAP or APbb... this is to distinguish it from non-bbox approach, such as Mask

