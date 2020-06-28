weight decay vs l2 regulatization

l2 regularization and weight decay are kind of the same thing

but should always use weight decay version



l2 vs weight decay

https://arxiv.org/abs/1706.05350


l2 regularization doesn't do anything if batch norm is being used




we take out input and multiply it by a weight matrix, then we get the activations
then we have batch norm, which is a big vector of adds and a big vector of multiplies (also does normalization, but these are the learned parameters)

let's say you then go into weight decay. let's say it's a large value...
then you multiple the weights by the weight decay????
that would create really high values which would destroy your loss function, but fortunately batch norm can fix it
but then, if batch norm fixes it by multiplying everything by that big number, the weights have to be divided by that big number. so the weight decay is canceled out.

seems like this assumes relu is before batch norm, because he's saying the activations go into batch norm

although weight decay clearly does something, as this paper exploresa: https://arxiv.org/abs/1810.12281
(here's the original paper saying it doesn't work: https://arxiv.org/pdf/1706.05350.pdf)







l2 vs weight decay

weight decay is usually defined as a term that’s added directly to the update rule. e.g., in the seminal AlexNet paper

there's a factor of 2 difference

note: some frameworks may define the L2 term with a 0.5 in front so that it cancels out the factor of 2 in the gradient, but that is not the case with keras.



the exact manner that a deep learning framework implements weight decay/regularization will actually affect what these solvers will do
