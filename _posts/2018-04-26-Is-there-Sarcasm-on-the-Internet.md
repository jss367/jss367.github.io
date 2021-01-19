---
layout: post
title: "Is there Sarcasm on the Internet?"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/elk.jpg"
tags: [Python, Scraping]
---
Is there sarcasm on the Internet? OK, that's an easy one. Here's a more difficult question: can an AI be trained to detect that sarcasm? The first step to training that algorithm would be to create a corpus of sarcastic statements from the Internet. Fortunately, there's a lot of sarcasm out there and, even more fortunately, much of it is already labeled.

 Both Reddit and Twitter have mechanisms where users can self-label a statement as sarcastic. On Reddit, users can end their statement with " /s" to denote sarcasm, such as: "That's a good idea /s". For Tweets, the #sarcasm hashtag is popular: "That's a good idea #sarcasm". Both of these forms of self-labeling make them a prime target to be used in a machine learning algorithm. In this notebook, I'm going to show how to scrape Reddit data and extract sarcastic statements.

The most popular way to scrape Reddit from Python is with [PRAW: the Python Reddit API Wrapper](https://praw.readthedocs.io/en/latest/). The first step is to download it.  Note that you can do so inside a Jupyter Notebook with the command: `!pip install praw`.

Reddit's API is easy to use and quick to sign up for. You will need a Reddit account. After you have an account, you will need to go to their [applications page](https://www.reddit.com/prefs/apps) and click on "create another app..." on the bottom left. Then enter a name for your application. Note the per the [Reddit API rules](https://www.reddit.com/wiki/api) you cannot use the word "reddit" in the title unless you have the word "for" preceding it. E.g. "scraper for reddit" is allowed but "reddit scraper" isn't. The rest of the application can be completed as follows:

* name: scraper
* [click on the "script" checkbox]
* description: A simple application for scraping data
* about url:
* redirect uri: http://localhost:8080  [see the praw documentation for more information]

You will get a 14-character personal use script and a 27-character secret key. You will need them to authenticate with Reddit.


```python
import praw
import re
```


If you are only collecting public comments, you won't need to include your username and password. If you are trying to access anything from your own account you will need to add it to the call below.


```python
data = praw.Reddit(client_id=my_client_id,
                  client_secret=my_client_secret,
                  user_agent='scraper')
```

Let's look in the AskReddit subreddit.


```python
subreddit  = data.subreddit("AskReddit")
```

Now we can look at the most popular threads on AskReddit.


```python
for submission in subreddit.top(limit=50):
    print(submission.title)
```

    With all of the negative headlines dominating the news these days, it can be difficult to spot signs of progress. What makes you optimistic about the future?
    How would you feel about a law that requires people over the age of 70 to pass a specialized driving test in order to continue driving?
    Professor Stephen Hawking has passed away at the age of 76
    [Breaking News] Orlando Nightclub mass-shooting.
    What are some slang terms a 50 year old dad can say to his daughter to embarrass her?
    What are the best free online certificates you can complete that will actually look good on a resume?
    How would you feel about a law requiring parents that receive child support to supply the court with proof of how the child support money is being spent?
    If they made a show called "White Mirror" that was about all the positive aspects of the human/technology relationship, what would be the plot of certain episodes?
    You have been accepted for an experiment: you must stay in a room with nothing but bed/toilet/food/water and no human contact for one month. If you succeed for the whole month without giving up, you get $5,000,000. Do you accept? And what are your coping strategies to avoid mental breakdown?
    What free software is so good you can't believe it's available for free?
    A British charity that helps victims of forced marriage recommends hiding a spoon in your underwear if your family is forcing you fly back to your old country, so that you get a chance to talk to authorities after metal detector goes off - have you or anyone else you know done this & how did it go?
    [Serious] Should elderly people be forced to take tests regarding their motor vehicle operating abilities and mental fitness and get their motor vehicle license(s) revoked if they fail the test(s)? Why/why not?
    People who do 30mph on an on ramp to a highway where the speed limit is 65mph. Why do you do this?
    If authors 'covered' novels, the way musicians cover songs, which covered novel would you be most excited to read?
    Blind gay people of Reddit, how did you know you were gay?
    Daughters of reddit, what is something you wish your father knew about girls when you were growing up?
    What bot accounts on reddit should people know about?
    Let's pretend violent video games teach you to use a gun to kill people. What other skills have you inconspicuously picked up playing video games?
    Would you continue to be vegan if you had to grow every single vegetable you wanted to eat? Why or why not?
    Redditors with less than a year left to live, what is on your bucket list and how can we help you?
    What tasty food would be distusting if eaten over rice?
    What’s a "Let that sink in" fun fact?
    What will be the "turns out cigarettes are bad for us" of our generation?
    People who made an impulse decision when they found out Hawaii was going to be nuked, what did you do and do you regret it?
    People who make passive-aggressive posts on /r/Askreddit that accomplish nothing, why do you do this?
    What are some good weird questions to ask someone to get to know them better?
    What's your most unbelievable "pics or it didn't happen" moment, whereby you actually have the pics to prove it happened?
    Your options are: 50 hawks, 10 crocodiles, 3 brown bears, 15 wolves, 1 hunter, 7 cape buffalo, 10,000 rats, 5 gorillas and 4 lions - you must pick 2 that will defend you while the rest are coming to kill you. Which do you pick and why?
    What YouTube channel is great to binge?
    [Serious]What are some of the creepiest declassified documents made available to the public?
    What's the fastest way you've seen someone improve their life?
    What's a short, clean joke that gets a laugh every time?
    What's a 10/10 album from the last 15 years by a relatively obscure artist/band?
    If your employer gave you the option to work 10 hr days Mon-Thurs instead of 8 hr days Mon-Fri would you do it? Why not why not?
    Redditors who have eaten at the Times Square Olive Garden, why?
    If you suddenly came into the possession of 20 tons of Nutella what would you do?
    What is the most interesting documentary you've ever watched?
    You wake up in Kim Jong Un's body. You can speak and understand Korean. Without getting assassinated by your commanders, how do you transition North Korea and its people from an Orwellian state of despair to a prosperous nation so you can then ride your fame to launch your career in music?
    What is your go-to never-fail joke?
    What is unethical as fuck, but is extremely common practice in the business world?
    Hawaii wants to create a law that will ban games with loot boxes to people under 21 years old. What do you think about that?
    [Breaking News] Donald Trump will be the 45th President of the United States
    Besides attacking McDonalds employees for sauce packets, whats the worst fan-boy meltdown you've seen in public?
    What is extremely rare but people think it’s very common?
    What common product has a feature you’re not sure everyone is aware of?
    When did your "Something is very wrong here" feeling turned out to be true?
    Multilingual Redditors, What is your "They didn't realize I spoke their language" story?
    What are some red flags for teachers that scream "drop this class immediately?"
    What is a dirty business tactic that you know and everyone should be aware of it?
    Non-Americans of Reddit, what's the biggest story in your country right now?
    

Now let's extract the ones with the sarcasm marker " /s"


```python
list_of_comments = []
# This takes a while to go through all the child comments
submission.comments.replace_more(limit=10)
for top_level_comment in submission.comments:
    list_of_comments.append(top_level_comment.body)
```


```python
for comment in list_of_comments:
    has_sacasm = re.search(' \/s', comment)
    if has_sacasm:
        print(comment)
        print("\n\n\n")
```

    Syrian, we have this little conflict going on, nothing too big /s 
    
    
    
    
    

This works, but it would take quite a lot to scrape enough data to have a sufficient dataset. Fortunately, much of the scraping has already been done. Someone has already scraped and made available [Reddit comments dating back to December 2005](http://files.pushshift.io/reddit/comments/).

After doing this, I realized that someone has already gone through that data and extracted sarcastic statements from it. They also created the largest corpus of sarcastic statements, containing 1.3 million sarcastic statements, which they call the [Self-Annotated Reddit Corpus (SARC)](http://nlp.cs.princeton.edu/SARC/). To read more about their corpus or to see their baseline detection rates, see their paper [A Large Self-Annotated Corpus for Sarcasm](https://arxiv.org/pdf/1704.05579.pdf).
