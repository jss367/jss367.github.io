

easy stuff:
was your  gpu empty when  you started?

hard stuff:

where does the memory go?1. compute the gradients2. backprop the gradients
Everything with learnable parameters needs to store it input until the backward pass- conv layers, fc layers, even batch norm

The amount of memory that is stored will change during training