pascal voc has bounding boxes like this:

Y value of top left corner
X value of top left corner
height
width

SO DOES COCO!!!     -- i think coco switches height and width

"bbox": [x,y,width,height],

i prefer dataclasses or namedtuples



fastai has functions that do:
The first thing we do is turning those into a more friendly format (calling get_trn_anno(); FYI bb_hw operates in the opposite direction, getting back to height-width format).

X value of top top left corner
Y value of top left corner
X value of bottom right corner
Y value of bottom right corner





http://cocodataset.org/#format-data



if there are no supercategories, just do none


