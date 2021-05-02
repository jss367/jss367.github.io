






5k samples

mAPs goes from 1.9 to 3.2, so 68% improvement (3.2-1.9) / 1.9

for 14k samples, see row three of table five

14000 6.8 17.5 23.9 16.4 9.5 22.1 29.8 19.9



mAP is not a linear combination of mAPs, m, l. It is computed by averaging AP over all categories, and different categories have different ratios of object sizes.

For example, let's say you're looking for birds and horses. Some of the birds are AP-s and some are AP-m. Also, some of horses are AP-m and some AP-l. 

AP is NOT weighted by classes. 



AP_s

small object - 100 examples

medium object - 10 examples

large object - 1 example

mAPs would be heavyily skewed


would be better to weight the different classes




