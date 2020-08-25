make sure you stick a __init__.py file everywhere



timing:

you should use timeit:
https://stackoverflow.com/questions/17579357/time-time-vs-timeit-timeit




    '{0}, {0}, and {1}'.format('ham', 'spam')
    

    can't do that with fstringsd = dict(name='example', description='This is an example', other=12, data=18)
    print('I catch kwargs... {name}: {description}'.format(**d))
    
    
    Another thing to think about when looking at the try/except paradigm is that you might have to ask for permission many times, but you only have to ask for foregiveness once. Take the simple example of opening a file. First, you have to check if the file exists. That's not hard to do. `os.path.exists` is all you need, or `.exists()` if pathlib is your thing. But then, if you're really being careful, you'll need to check if you have permissions. Then, what if something goes wrong when reading the file? The encoding isn't what you expect or it's corrupted or something else.
    
   That's why try/except is nice. No matter what the issue was, you can catch the exception and deal with it gracefully..
   
   
   BaseException is a bad idea because it catches ALL exceptions, even those you don't want to catch, like KeyboardInterrupt.
   
   It's even mentioned in Python: https://docs.python.org/3/glossary.html#term-eafp
   
   
   
