


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


