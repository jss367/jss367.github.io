---
layout: post
title: "Debugging Python List Comprehensions"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/tree_sunset.jpg"
tags: [Python]
---

I ran into an interesting issue when debugging in Python that I thought was worth sharing. It came up when I was visualizing some results from an object detector. I had a class for an object detector and one of the things it would check was that each prediction was associated with a valid object in the object dictionary. Here is a mock-up of the relevant parts (this is not the real class, just a toy example so as not to distract from the point).

```python
class ObjectDetector():
    def __init__(self, object_map):
        self.object_map = object_map
        self.predictions = self.get_predictions()
        assert all([pred in self.object_map for pred in self.predictions])
        
    def get_predictions(self):
        "Spoofed result"
        return [1, 1, 1, 1, 1, 2, 2, 2, 3, 3]

object_map = {0: 'cat', 1: 'dog', 2: 'bird'}
od = ObjectDetector(object_map)
print(od.predictions)
```

After running the code, I got the following error:

```python
$ python object_detector.py
Traceback (most recent call last):
  File "object_detector.py", line 13, in <module>
    od = ObjectDetector(object_map)
  File "object_detector.py", line 6, in __init__
    assert all([pred in self.object_map for pred in self.predictions])
AssertionError
```

Interesting. I wasn't sure what the problem was so I decided to drop in a debugger. Immediately before the assert statement, I dropped in `import ipdb; ipdb.set_trace()`.

When I debug, I like to go nice and easy. Step 1 is to recreate the error message and take it from there. Actually, I should call it step 0 because it couldn't possibly go wrong (yes, you know what's going to happen next).

```python
ipdb> assert all([pred in self.object_map for pred in self.predictions])
*** NameError: name 'self' is not defined
```

What? I wanted an `AssertionError`, not a `NameError`. How did that happen? Where did `self` go?

```python

ipdb> self
<__main__.ObjectDetector object at 0x00000220AB590CC8>
ipdb> self.object_map
{0: 'cat', 1: 'dog', 2: 'bird'}
```

Oh, it's right where I left it. So what happened before? Maybe I'm missing something else? Let's see if this works:

```python
ipdb> self.predictions[0] in self.object_map
True
```

OK, all looks good. Now I'll just...

```python
ipdb> [pred in self.object_map for pred in self.predictions]
*** NameError: name 'self' is not defined
```

Arrgh! I was trying to sneak that one by. I was pretty stuck at this point, but I'll spare you the head-scratching and the old standby: turn-it-off-and-turn-it-back-on-again. The key here is to take a look at `globals` and `locals`.

```python
ipdb> globals()
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x00000220AB28D488>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'object_detector.py', '__cached__': None, 'ObjectDetector': <class '__main__.ObjectDetector'>, 'object_map': {0: 'cat', 1: 'dog', 2: 'bird'}}


ipdb> locals()
{'object_map': {0: 'cat', 1: 'dog', 2: 'bird'}, 'ipdb': <module 'ipdb' from 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\ipdb\\__init__.py'>, 'self': <__main__.ObjectDetector object at 0x00000220AB590CC8>}
```

That's odd... why is `self` in `locals` but not in `globals`? It should be in `globals`. Fortunately, I can add it with...

```python
globals().update(locals())
```

Then check it again...

```python
ipdb> globals()
{'__name__': '__main__', '__doc__': None, '__package__': None, '__loader__': <_frozen_importlib_external.SourceFileLoader object at 0x00000220AB28D488>, '__spec__': None, '__annotations__': {}, '__builtins__': <module 'builtins' (built-in)>, '__file__': 'object_detector.py', '__cached__': None, 'ObjectDetector': <class '__main__.ObjectDetector'>, 'object_map': {0: 'cat', 1: 'dog', 2: 'bird'}, 'ipdb': <module 'ipdb' from 'C:\\Users\\Julius\\anaconda3\\envs\\tf\\lib\\site-packages\\ipdb\\__init__.py'>, 'self': <__main__.ObjectDetector object at 0x00000220AB590CC8>}
```

There it is. Now we can...

```python
ipdb> [pred in self.object_map for pred in self.predictions]
[True, True, True, True, True, True, True, True, False, False]
```
Ah, that's better. From here on out it's simple to debug.

```python
ipdb> self.predictions
[1, 1, 1, 1, 1, 2, 2, 2, 3, 3]
```

Hmm... it's the "3"s that are the problem.

```python
ipdb> self.object_map.keys()
dict_keys([0, 1, 2])
```

D'oh! I 0-indexed the objects in the `object_map` but not in the `predictions`. The original bug was a simple one and easy to fix, but the missing `self` definitely confounded me for a bit.

I found (afterwards, of course) that this had been [reported way back in 2010](https://github.com/ipython/ipython/issues/62)! It's been reported a [few times](https://github.com/ipython/ipython/issues/136) to [different debuggers](https://github.com/inducer/pudb/issues/103) and is an issue for dictionary comprehensions as well. There's a bit of a deeper discussion in the threads if you're interested, but, in summary, it's actually a Python issue, and not specific to any debugger. And based on how long it's been open, I don't expect it to be fixed any time soon. Just remember, `globals().update(locals())` is your friend.
