



Thoughts on training models




I'm pretty convinced that you should use transfer learning whenever you can, and you almost always can.

Any kind of image, I would transfer learn. THe first filters are mostly color separators and Gabor filters (link to distil paper). You're going to need them no matter what. Even for geospatial analytics, I use transfer learning. Use it all the time.


Use some form of normalization. If you have large batch size, use Batch norm. If small, maybe try group norm or isntance norm (this says instance norm: https://towardsdatascience.com/a-bunch-of-tips-and-tricks-for-training-deep-neural-networks-3ca24c31ddc8)?




Overfit first. If you can't overfit, you can't succeed. If you can't, you need a bigger network or cleaner data or something.

You can always use data augmentation, L1, L2, or dropout to regularlize if you need to

You can also contrain your weights directly (model.add(Conv2D(64, kernel_constraint=max_norm(2.)))) but I recommend weight decay (better than L2?)



shuffle your training data


