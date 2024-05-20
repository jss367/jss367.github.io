
data preprocessing is going to be different for different models

neural networks can create embeddings for categorical variables, which are often better than one-hot encoded

for xgboost, don't think you need to one-hot
--- well, if it's not ordered data, you might want to
the downside is that you take a compact variable and spread it into a bunch of sparse variables

I generally still one-hot, but sometimes you try both



cat boost works well with less preprocessing
