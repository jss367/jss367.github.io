The biggest problem with reading scientific papers is that you lack context.


Sometimes the context is obvious. If you're reading Yolov3 you're expected to have read Yolo and Yolo:9000. a



the way dropout was implemented in the paper isn't the way it's actually done in pytorch or other frameworks. see 
https://course.fast.ai/videos/?lesson=6 - 37 minutes (2019 version)


some are completely wrong, like how batch norm actually helps



some are really hard to understand
- great researchers are not necessarily great explainers. Some are, like Carl Sagan. That's rare.


some aren't that hard, but you walk away with a two-sentence concept after reading pages and pages
For example, DenseNets are like ResNets except that instead of adding the skip connection you concatenate it
If you reading the paper much later, you don't need to have proof that it works - so many other people in the community have used it that's your proof.



If you don't know the symbols, this wikipedia list is good: https://en.wikipedia.org/wiki/List_of_mathematical_symbols