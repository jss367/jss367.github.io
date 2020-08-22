Optimizers:


I keep getting lost in the world of optimizers.

All of a sudden adam is bad and there's something better. Then shortly after no one is talking about that new thing anymore and 


# Overview

adam - Adaptive Moments
- why increase each parameter the same amount?

adagrad
adadelta
rmsprop
sgd + momentum


Adaptive optimizers help with faster convergence.
SGD + momentum requires good initialization and takes longer, but can be better (this site says find global minimum: https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8)





ResNest:
pose estimation experiment: We use Adam optimizer with batch size 32 and initial
learning rate 0.001 with no weight decay
- no other mentions of optimizers




[DETR](https://arxiv.org/abs/2005.12872) was trained with AdamW. This paper also notes that "Transformers are typically trained with Adam or Adagrad optimizers with verylong training schedules and dropout"
- this paper also says that "FasterR-CNN, however, is trained with SGD with minimal data augmentation andwe are not aware of successful applications of Adam or dropout"





