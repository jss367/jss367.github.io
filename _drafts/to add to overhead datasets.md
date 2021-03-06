https://github.com/chrieke/awesome-satellite-imagery-datasets

add additional categories to dota

Having robust benchmarking datasets is a critical 

Benchmarking datasets are hugely important to the development of neural networks, and ImageNet in particular is significant in the improvements in computer vision over the last five years. Given the importance of benchmarking datasets, is it any wonder that domains without that don't seem to cultivate innovation at the same levels? In particular, geospatial analytics doesn't have nearly the level of new networks that one would imagine given the importance, and hopefully the DOTA dataset can change that.

First, why does geospatial analytics even need it's own dataset? Can't you just fine-tune off of ImageNet weights? Well, yes, you can, and you can even get pretty far that way. But there are important differences between overhead images and the "normal" images we see in ImageNet or a similar benchmarking dataset.

Here are some of the key differences between overhead and natural imagery:

* Arbitrary orientations
** These are significant factors because they influence the architectures that will be developed. Many architectures (YOLO, Faster-RCNN) have a bias towards horizontally oriented objects. This is not acceptable when in overhead imagery.
* Greater scale variation
** The scale matters because anchor sized are often-prechoosen and can't be easily changed.
* Objects are more densely packed cluttered
** large aspect ratios - bridges, ships, roads, etc.
** The denser packed objects make NMS more difficult
* More class imbalance
** possibly because real life is more unbalanced too but the public datasets correct for this. but it's harder to correct in overhead so they don't

way more instances per image
PASCAL VOC - 2.89 instances per image
MSCOCO - 7.19 (they have slightly different numbers elsewhere, good to check?)
ImageNet - 1.37
DOTA - 67.10


Large datasets are especially important for geospatial analytics, because most algorithms are pre-trained from natural images, such as ImageNet and MS COCO.


There is a need for overhead imagery datasets. Natural imagery doesn't have the same distributions.


what to judge a dataset on:
 1) a large number of images, 2) many instances per categories, 3) properly oriented object annotation, and
4) many different classes of objects, which make it approach to real-world applications.




Dataset 					Annotation way 	#main categories 	#Instances 	#Images 	Image width
NWPU VHR-10 [2] 			horizontal BB 	10 					3651 		800			∼1000
SZTAKI-INRIA [1] 			oriented BB 	1 					665 		9 			∼800
TAS [9] 					horizontal BB 	1					1319		30			792
COWC [20] 					one dot 		1 					32716 		53 			2000∼19,000
VEDAI [24] 					oriented BB 	3 					2950 		1268 		512, 1024
UCAS-AOD [39] 				oriented BB 	2 					14,596 		1510 		∼1000
HRSC2016 [17] 				oriented BB 	1 					2976 		1061 		∼1100
3K Vehicle Detection [15] 	oriented BB 	2 					14,235 		20 			5616
DOTA 						oriented BB 	14					188,282 	2806 		800∼4000

More notes:




DOTA
introduce a large-scale Dataset for Object deTection in Aerial images (DOTA)
- 2806 aerial images from
different sensors and platforms. Each image is of the size about 4000 × 4000 pixels and contains
objects exhibiting a wide variety of scales, orientations, and shapes.
-  15 common object categories



Most datasets are from Google Earth imagery, so consist of RGB.



Datasets like TAS [9], VEDAI [24], COWC [20] and DLR 3K Munich Vehicle [15] only focus on vehicles. UCAS-AOD [39] contains vehicles and planes while HRSC2016 [17] only contains ships even though fine-grained category information are given. 










To add to overhead datasets:

image sizes 
* max
* min
* histogram of shapes


Add example images!!!



Training example frequency is low versus other disciplines. Few datasets exist that have appropriate labels
for objects within satellite imagery. The most notable
are: SpaceNet [38], A Large-scale Dataset for Object
DeTection in Aerial Images (DOTA) [40], Cars Overhead With Context (COWC) [27], and xView [18].



https://captain-whu.github.io/iSAID/


This post focuses on object detection datasets.


The xView Dataset [18] was chosen for the application of super-resolution techniques and the quantification
of object detection performance. Imagery consists of 1,415
km2 of DigitalGlobe WorldView-3 pan-sharpened RGB imagery at 30 cm native GSD resolution spread across 56 distinct global locations and 6 continents (sans Antarctica).
The labeled dataset for object detection contains 1 million object instances across 60 classes annotated with bounding boxes, including various types of buildings, vehicles,
planes, trains, and boats.
Unfortunately, many objects are
mislabeled or simply missed by labelers (see Figure 1).
In addition, many xView classes have a very
low number of training examples (e.g. Truck w/Liquid
has only 149 examples) that are poorly differentiated from
similar classes (e.g. Truck w/Box has 3653 examples and
looks very similar to Truck w/Liquid).



Datasets like TAS [9], VEDAI [24], COWC [20] and DLR 3K Munich Vehicle [15] only focus on vehicles. UCAS-AOD [39] contains vehicles and planes while HRSC2016 [17] only contains ships even though fine-grained category information are given. 

what to judge a dataset on:
 1) a large
number of images, 2) many instances per categories, 3) properly oriented object annotation, and
4) many different classes of objects, which make it approach to real-world applications.



https://medium.com/the-downlinq/spacenet-5-dataset-release-80bd82d7c528



DOTA
- is there a variety of shapes?
- boxes are arbitrary quadrilateral


NWPUVHR
- annotated by experts


DIOR - object DetectIon in Optical Remote sensing images

"DIOR" is a large-scale benchmark data set proposed for object detection in optical remote sensing images
 


Other notes:

Currently no pixel level overhead imagery datasets for object detection


another dataset:
LEVIR [91] 2018 Consists of ∼22,000 Google Earth images and ∼10,000 independently labeled
targets (airplane, ship, oil-pot). url: https://pan.baidu.com/s/1geTwAVD


DLR3K [87] 2013 The most frequently used datasets for small vehicle detection. Consists of
9,300 cars and 160 trucks. url: https://www.dlr.de/eoc/en/desktopdefault.aspx/
tabid-5431/9230 read-42467/


## UCAS-AOD
This is a dataset by the University of Chinese Academy of Sciences

Here's the website: https://ucassdl.cn/
2 categories: Airplanes, Vehicles
910

UCAS-AOD contains 1,510 satellite images (≈ 700 × 1300 px) with 14,595 objects annotated by OBBs for two categories:. The dataset was randomly split into 1,110 training and 400 testing images.


