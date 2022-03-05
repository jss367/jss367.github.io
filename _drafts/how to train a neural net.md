


How to train a neural network

The rule of thumb for determining the embedding size is the cardinality size divided by 2, but no bigger than 50.





increasing the width,depth, and resolution of the model in sync
- Efficientnet: Rethinking modelscaling for convolutional neural networks



deeper architectures that require more regularization




If you're using one-cycle-training (fit one cycle, cyclic learning rates, etc.), it's better not to use EarlyStopping. That's because it will stop with it's learning rate in a weird place. If you were able to lower its learning rate a bit, you'd like find a better optimal solution. Thus it's better to pick the number of epochs that you'll actually use.

In fact, because of double descent, you might not want to use early stopping at all.



Use early stopping if you are only gently decaying your LR


visualizing

Grad CAM
heatmaps


Other tricks:


Label Smoothing,
MixUp and other data aug
Half Precision



if an image is too large (for im class), use a different random crop each epoch


go through the data

look for issues - i've already blogged about how to look through a dataset
* noisy
* images where you can't tell the answer
* 


start with a small, end-to-end pipeline

I usually do no augmentation, because I want to see the performance improvement.

small model (e.g. resnet 18 or 34 backbone)


use your default optimizer (mine is adam)

do think carefully about your metric. It doesn't have to be perfect right away. you can always add things like focal loss if you think you're having a problem with unbalanced data, but it should at least make sense. start with a simple one. if sem seg, maybe iou or dice.



number one debugging:
- it's almost always the data. look at the data right before it enters your model

use tensorboard. develop a sense of what training loss should look like. turn off smoothing (it lies)

overfit on a small subset of the data. if you can't overfit, you're not going to be able to classify it.

tips to overfit
* train for a really long time
* use a smaller and smaller subset of the data until you can overfit
* use a larger model

then reduce overfitting

allow more data
more augmentation
more regularization
smaller model (last choice, you should still be able to train a somewhat large model, say resnet 50/xception size, on most datasets)




