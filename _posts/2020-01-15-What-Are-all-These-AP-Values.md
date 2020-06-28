---
layout: post
title: "What Are all These Different AP Values"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/palm_sunset.jpg"
tags: [Python]
---

The most frequently used evaluation metric for object detection is "Average Precision (AP)", which was originally introduced in [The PASCAL Visual Object Classes Challenge 2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/) (VOC2007). It's a relatively simple concept, but when you look at research papers, you'll often see lots of variations on the basic concept. This post aims to clarify those variations.

## Precision and Recall

Before we can get into average precision, we'll have to start with **precision** and **recall**. These are two of the most important metrics when evaluating the quality of a statistical model. They are rigorous statistical methods with intuitive meanings. Precisions asks, "Out of all the times the model said it was positive, what percentage were correct?" Recall, asks "Out of all the times there was a positive, what percentage were found by the model?" Mathematically, they are calculated by looking at the number of true positives, false positives, and false negatives as follows:

$$ \text{Precision}=\frac{TP}{TP+FP} $$

$$ \text{Recall}=\frac{TP}{TP+FN} $$

Note that the notion of "true negative" doesn't exist for object detection. Every group of pixels that's not an object would be a negative so there are too many for the metric to be meaningful.

## Thresholds and IoU

The notion of a **true positive** sounds fairly intuitive, but there are actually some important details to work out. If a model predicts a dog in a photograph and the bounding box is off by a few pixels, does it still count? To determine that, we need a threshold for how much overlap to allow. The metric we use for this is known as the **Intersection over Union**, or **IoU**, score. It is defined as follows:

$$ IoU(A,B) = \frac{A \cap B}{|A \cup B|} $$

The IoU score is also known as the **Jaccard index**.

## Average Precision

Now let's turn to **average precision**. The term is exactly what it sounds - the average precision, as defined above, of a model. What are we averaging? Well, by varying the threshold of the IoU score, we will produce different values of precision and recall. Thus the average precision is calculate by sampling the precision at a bunch of different thresholds and averaging them.

So, how do you calculate it? You have to change the threshold to keep increasing the recall, then find the precision at each point. Note that if you actually do this precision doesn't decrease monotonically. That's OK. You make the curve monotonic by setting the precision to be the highest value at that given recall or higher. This is because if you land at a bad spot you could always tweak the threshold to a value that's going to give you higher precision and recall.

OK, so you know that "Average Precision", or AP, is the area under the precision recall curve. AP is between 0 and 100, with higher numbers better. Now let's go one step further. 

## Multiclass classification

Many datasets have lots of objects. [COCO](http://cocodataset.org/#home), for example, has 80. To compare performance over all object categories, the **mean AP (mAP)** averaged over all object categories is usually used. This means we find the AP value for each class and take the average. Note that mAP is not weighed by category frequency. If you have 1000 examples of a cat and an AP value of 0.9 and 10 examples of a dog and an AP value of 0.7, your mAP would be 0.8.



Sometimes, different competitions use different values. For Pascal VOC, you'll often see mAP@0.5. For COCO, you'll see mAP@[0.5:0.95]. This is a third average!




What does map@(0.5:0.95)mean? 


To measure the object localization accuracy, the Intersection over Union (IoU) is used to check whether the IoU between the predicted box and the ground truth box is greater than a predefined threshold, say, 0.5. If yes, the object will be identified as “successfully detected”, otherwise will be identified as “missed”. 

The 0.5- IoU based mAP has then become the de facto metric for object detection problems for years.

But these are really different things. The threshold you use will influence which types of models appear to perform best. 0.5 threshold is very coarse localization while 0.95 is very precise.

## COCO AP
Instead of using a fixed IoU threshold, MS-COCO AP is averaged over multiple IoU thresholds between 0.5 (coarse localization) and 0.95 (perfect localization). This change of the metric has encouraged more accurate object localization and may be of great importance for some real-world applications

In COCO they change the IoU values from 50% to 95%, at a step of 5%.

# Examples

OK, let's look at some examples. Here's a table from [Cascade R-CNN](https://arxiv.org/abs/1906.09756):


![metric]({{site.baseurl}}/assets/img/metrics/cascade_rcnn.png "Metrics")



Here are some results from Faster R-CNN:

![metric]({{site.baseurl}}/assets/img/metrics/faster_rcnn.png "Metrics")


It's showing mAP values at a threshold of 0.5, then the average of values at all thresholds from 0.5 to 0.95 and steps of 0.05.


Sometimes you'll see APbbox or BoxAP or APbb... this is to distinguish it from non-bbox approach, such as Mask
Sometimes you'll see box AP. It's the same as AP but they're highlighting that it's for bounding boxes. You'll also see this written as APbb, short for "AP bounding box". Here's an example from [Mask R-CNN](https://arxiv.org/abs/1703.06870):

![metric]({{site.baseurl}}/assets/img/metrics/mask_rcnn.png "Metrics")


## Other attempts

There are other approaches but they haven’t taken off: https://arxiv.org/pdf/1807.01696.pdf


What is Box AP? - I think it’s the same as AP. I guess it just means with bounding boxes. As opposed to with Masks
“The proposed methods are only applied on the detection branch in Mask R-CNN. APbb means the detection performance and APmask indicates the segmentation performance”
There’s also APs, APm, APl - small, medium, large
Maybe look at example from cascade mask rcnn or something



So you get 10 different values.. The mean of those values is the AP@[0.5:0.95].

There is not perfect consistency here. There is very good consistency, especially when benchmarking on datasets like COCO. 







# Small Objects

ap 50:95 is not a good threshold for small objects... one pixel off causes quite a lot of loss

# Other




# F1

Better for production because you have an actual threshold value.

$$ F1 = 2 \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{ \mathrm{precision} + \mathrm{recall}} $$


Note that this is a generalization of FBeta

$$ F_\beta = (1 + \beta^2) \cdot \frac{\mathrm{precision} \cdot \mathrm{recall}}{(\beta^2 \cdot \mathrm{precision}) + \mathrm{recall}} $$
