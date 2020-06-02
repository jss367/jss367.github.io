Quick wins in Machine Learning

Rectified Adam?


Some of these are specific to object detection:

Soft NMS: http://www.cs.umd.edu/~bharat/snms.pdf

leaky relu

warm up strategy:
https://openreview.net/pdf?id=r14EOsCqKX

Not freezing batch norm laters. From FastAI:  There was only one approach that consistently worked well across all datasets that wetried, which is to never freeze batch-normalization layers, and never turn off the updating of their movingaverage statistics. Therefore, by default,Learnerwill bypass batch-normalization layers when a user asksto freeze some parameter groups. Users often report that this one minor tweak dramatically improvestheir model accuracy and is not something that is found in any other libraries that we are aware of.
