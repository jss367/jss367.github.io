---
layout: post
title: Overhead Imagery Datasets for Object Detection
description: "A chart with detailed information about overhead imagery datasets for object detection"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/gosse_bluff.jpg"
tags: [Geospatial Analytics]
---

This post provides a summary of some of the most important overhead imagery datasets. I intend to make this a living document that is continuously updated.

<b>Table of contents</b>
* TOC
{:toc}

# Overhead Imagery Datasets Overview

|Dataset Name | 										Total Number of Objects | 	Number of Images | 	Number of Categories | Image Size | 			Resolution | 		Annotation Type | 			Source |				Year Released| Restrictions
|------------------ | --------------------- | --------------------- | ---------------------
|[DIOR](http://www.escience.cn/people/gongcheng/DIOR.html) | 			192,472 | 					23,463 | 				20 | 				large |	TBD				 | 	Horizontal Bounding Boxes | 	TBD |					2020	| None
|[DOTA](https://captain-whu.github.io/DOTA/dataset.html) | 			188,282 | 					2,806 | 				15 | 				387 X 455 - 4096 X 7168 |	most 20-40cm (see below) | 	Rotated and Horizontal Bounding Boxes | 	Google Earth (mostly) and satellites |					2018	| Academic purposes only; any commercial use is prohibited
|[XVIEW](http://xviewdataset.org/) | 1,000,000+ | 	TBD | 			60 |				large |						30 cm | 		Horizontal Bounding Boxes |	WorldView-3 satellites | 2018 | Non-commercial use
|[NWPU VHR-10](http://www.escience.cn/people/gongcheng/NWPU-VHR-10.html) | 		3,651 | 						800 | 				10 | 				large | 						8-200cm | 		Horizontal Bounding Boxes | Google Earth and Vaihingen dataset | 2016 | Research purposes only
|[COWC](https://gdo152.llnl.gov/cowc/) | 			32,716 |					TBD |				1 (cars) |			large |						15 cm | Center Points | 			TBD | 2016 | None

# Descriptions

## DIOR

DIOR is a huge dataset with ten times the number of images as DOTA, although a similar number of objects. It is the most recent dataset on the list.

#### Academic paper

[Object detection in optical remote sensing images: A survey and a new benchmark](https://www.sciencedirect.com/science/article/pii/S0924271619302825)

#### Categories

Airplane, Airport, Baseball field, Basketball court, Bridge, Chimney, Dam, Expressway service area, Expressway toll station, Harbor, Golf course, Ground track field, Overpass, Ship, Stadium, Storage tank, Tennis court, Train station, Vehicle, and Windmill


## DOTA

DOTA is a large dataset that combines aerial and satellite imagery. It combines different sensors and platforms.

#### Academic paper

[DOTA: A Large-scale Dataset for Object Detection in Aerial Images](https://arxiv.org/abs/1711.10398)

#### Categories
(abbreviations used on leaderboard are shown in parentheses)

Plane, Ship, Storage Tank (ST), Baseball Diamond (BD), Tennis Court (TC), Basketball Court (BC), Ground Track Field (GTF), Harbor, Bridge, Large Vehicle (LV), Small Vehicle (SV), Helicopter (HC), Roundabout (RA), Soccer Ball Field (SBF), Basketball Court

#### Leaderboard

[DOTA Leaderboard](https://captain-whu.github.io/DOTA/results.html)

#### GSD Distribution

Here are histograms and boxplots of the ground sample distance for images in the dataset (when provided). Outliers have been excluded from the box plot for clarity.

![DOTA_GSD]({{site.baseurl}}/assets/img/Training_histo.png "DOTA GSD")

![DOTA_GSD]({{site.baseurl}}/assets/img/train_box_plot.png "DOTA GSD")

![DOTA_GSD]({{site.baseurl}}/assets/img/Validation_histo.png "DOTA GSD")

![DOTA_GSD]({{site.baseurl}}/assets/img/val_box_plot.png "DOTA GSD")

## xView
The xView dataset contains over 1 million objects across 60 classes covering over 1,400 km^2. Objects in xView vary in size from 3 meters (10 pixels) to greater than 3,000 meters (10,000 pixels).


#### Academic paper

[xView: Objects in Context in Overhead Imagery](https://arxiv.org/abs/1802.07856)

#### Categories
They use ontological labels, which I like. Although for datasets that don't, this same idea could be done after the fact.

Aircraft Hangar, Barge, Building, Bus, Cargo Truck, Cargo/container Car, Cement Mixer, Construction Site, Container Crane, Container Ship, Crane Truck, Damaged/demolished Building, Dump Truck, Engineering Vehicle, Excavator, Facility, Ferry, Fishing Vessel, Fixed-wing Aircraft, Flat Car, Front Loader/bulldozer, Ground Grader, Haul Truck, Helicopter, Helipad, Hut/tent, Locomotive, Maritime Vessel, Mobile Crane, Motorboat, Oil Tanker, Passenger Vehicle, Passenger Car, Passenger/cargo Plane, Pickup Truck, Pylon, Railway Vehicle, Reach Stacker, Sailboat, Shed, Shipping Container, Shipping Container Lot, Small Aircraft, Small Car, Storage Tank, Straddle Carrier, Tank Car, Tower, Tower Crane, Tractor, Trailer, Truck, Truck Tractor, Truck Tractor W/ Box Trailer, Truck Tractor W/ Flatbed Trailer, Truck Tractor W/ Liquid Tank, Tugboat, Utility Truck, Vehicle Lot, Yacht

##### Ontology

![xview_classes]({{site.baseurl}}/assets/img/xview_classes.jpg "xView Categories")

#### Label Noise

They do three stages of quality control, including a mix of manual and automated checks and a comparison with a gold standard (hand-annotated by experts) dataset. In order to pass expert quality control, the batch was required to have a precision of 0.75 and recall of 0.95 at 0.5 intersection over union (IoU) when compared to the gold standard.

However, despite these efforts the xView dataset still has considerable noise. I think that part of the problem is that a 0.75 precision requirement for ground truth data isn't very high. The [winning solution](https://arxiv.org/pdf/1903.01347.pdf) on the xView dataset challenge noted that using focal loss became problematic because "these exponentially higher weights lead to an extreme effect of hard and mislabeled samples". It seemed like part of their solution was of a loss function that worked well for messy imagery. Other researchers have noted that the [mislabeled data affected their model performance](https://insights.sei.cmu.edu/sei_blog/2019/01/deep-learning-and-satellite-imagery-diux-xview-challenge.html) as well.



#### Other
This dataset does well for geographic diversity.

![xview_distribution]({{site.baseurl}}/assets/img/xview_geographic_distribution.png "xView Distribution")

The images in this dataset, like most satellite images, were preprocessed by performing orthorectification, pan-sharpening, and atmospheric correction.

This dataset was released under a noncommercial license. See the [xView dataset rules](https://challenge.xviewdataset.org/rules) for more information.


## NWPU VHR-10
Northwestern Polytechnical University Very High Resolution-10

#### Academic papers

[Multi-class geospatial object detection and geographic image classification based on collection of part detectors](https://www.sciencedirect.com/science/article/abs/pii/S0924271614002524) (Paywall)

[A survey on object detection in optical remote sensing images](https://www.sciencedirect.com/science/article/abs/pii/S0924271616300144) (Paywall)

[Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images](https://ieeexplore.ieee.org/document/7560644) (Paywall)

#### Categories
Airplane, Ship, Storage Tank, Baseball Diamond, Tennis Court, Basketball Court, Ground Track Field, Harbor, Bridge, Vehicle

#### Label Noise

According to the website this dataset was manually annotated by experts, so the noise should be low.

#### Other

150 of the 800 images are background only (no objects).

These images are from Google Earth and Vaihingen data set. The Vaihingen data was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF).

## COWC

Cars Overhead With Context

#### Academic paper

[A Large Contextual Dataset for Classification, Detection and Counting of Cars with Deep Learning](https://gdo152.llnl.gov/cowc/mundhenk_et_al_eccv_2016.pdf)

#### Categories

Cars

#### Other

Data is from six different locations:
* Toronto, Canada
* Selwyn, New Zealand
* Potsdam, Germany
* Vaihingen, Germany
* Columbus, Ohio
* Utah

The imagery from Vaihingen, Germany and Columbus, Ohio is in grayscale.
