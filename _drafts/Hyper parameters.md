Hyper parameters

start with the defaults - they are pretty good.


here's is a list of some defaults, but use whatever the framework defaults are


epochs - also seeing iterations used a lot, such as detectron2 and the code for fyzer retinanet

batch size

loss function - is this a hyperparameter?

optimizer - sgd with momentum and adam are the most popular


learning rate - learn rate schedules are becoming popular, and they are great.
sometimes people will reduce learning rate by 10X for the last, say, 50,000 iterations. and again for the last 20,000 iterations.


