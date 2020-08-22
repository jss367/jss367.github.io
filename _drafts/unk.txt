Why do you need more than one? enc/dec


Share the weight matrix between embedding layers
So embeddings have both languages?



walk through
Source sentence goes in
Words become vectors based on some pre-trained embedding (such as GloVe)
Multi-headed attention
When translating this word, what words should I pay attention to?
Do this six times
Take the words that we’ve generated so far



How to self-attenion

Create three vectors from input vector
Query vector, key vector, and value vector
Size 64, although doesn’t have to be
So we calculate the self-attention for each word
We take that word’s query vector and take the dot product with every key vector (including its own)
Then scale by dividing by the sqrt of the length of the key vector
Leads to more stable gradients
Then pass through softmax to normalize
The word is going to have the largest result, which makes sense, but the words around it also affect it
Then take each value vector and multiply it by its softmax score
This decreases the significance of insignificant words
Then sum up the weighted value vectors
For just this word? Not clear what’s being summed here
This is now the output of the self attention layer for the first word
Send this along to the feed-forward network




Teacher-forcing



Is this were you determine which definition of “check” is being used?
The QKV matrices are different depending on if in encoder, decoder, or in between?


At one point V and Q are same, other points not the same


