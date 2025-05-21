---
layout: post
title: "Schrödinger's Cat"
description: "A simple quantum physics thought experiment showing that the notion of macroscopic objects (like cats) being simultaneously dead and alive does not withstand close inspection"
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/fishing_cat.jpg"
tags: [Python, Physics]
---

The story of Schrödinger's cat, a cat that through quantum physics is simultaneously alive and dead, has become engrained in popular culture and many popular science articles. But as the physics behind it has become popularized, misconceptions have been introduced into the story. When you Google "Schrodinger's cat" the following definition appears: "a cat imagined as being enclosed in a box with a radioactive source and a poison that will be released when the source (unpredictably) emits radiation, the cat being considered (according to quantum mechanics) to be simultaneously both dead and alive until the box is opened and the cat observed." The notion that a cat can be both alive and dead at the same time is counterintuitive and, most importantly, completely false. I propose a thought experiment that demonstrates that the popular conception of a half-living cat is impossible.<!--more-->


Let's assume we have the typical Schrödinger's cat experiment. We'll let the experiment run until the probability that the vial of poison is destroyed and therefore the cat is dead is 50%. After that time the vial will be removed and we will attempt to determine whether the cat is dead or alive.

Now let's set up the rest of the experiment. Attached to the experimental device is a printer. The printer knows whether the cat is alive or dead, but instead of just printing the results, it throws a little randomness into the mix. For its first printout, it will print either "The cat is alive" or "The cat is dead" with a 50% probability. In this case, the printout isn't in any way useful for determining the status of the cat. But for each printout after the first, the probability that the printout will be accurate increases by 1%. Thus the first will be accurate 50% of the time, the 11th printout 60% of the time, and the 26th will be accurate 75% of the time.

OK, now we'll got our experiment and our printer set up, let's run the experiment!


After running the experiment the vial is removed and the cat is either alive or dead, but we do not know. We'll print our first result,  which, because it's only accurate 50% of the time, will contain no actual information about the state of the cat. Here is the first printout:

    
    Trial number 1, where the printout has a 50% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    

Now the experiment has been conducted and a result has been printed, although the result is meaningless. Now we'll start with the real trials. For the first trial, the printout will be accurate 51% of the time. Thus the next printout will be the truth with a probability of 51% and a lie with a probability of 49%. And the one after that will have a 52% chance of being true, and so on.


    Trial number 2, where the printout has a 51% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 3, where the printout has a 52% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 4, where the printout has a 53% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    Trial number 5, where the printout has a 54% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 6, where the printout has a 55% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    

If it so happens that the person receiving the printout, we'll call her Alice, is a statistician. With her background, she is able to quickly calculate the true probability, based on the information she has, that the cat is alive.


    Alice calculates that the probability that the cat survived the experiment is 50.98%
    

Then we add a second person to the experiment. We'll call him Bob. Bob never sees or communicates with Alice, but he does get a printout from the experiment. However, instead of printing "The cat is alive" or "The cat is dead", it prints either "That statement was true" or "That statement was false". Now, Bob knows whether the statement that Alice is reading is true or false, but he does not know what Alice is acutally reading, and therefore doesn't know anything about the cat.

Let's continue with trials 11-20. But instead of seeing what Alice sees, we'll see only what Bob sees.


    Trial number 6, where the printout has a 55% chance of being accurate.
    Bob's printout says:
        That statement was true
    
    Trial number 7, where the printout has a 56% chance of being accurate.
    Bob's printout says:
        That statement was false
    
    Trial number 8, where the printout has a 57% chance of being accurate.
    Bob's printout says:
        That statement was false
    
    Trial number 9, where the printout has a 58% chance of being accurate.
    Bob's printout says:
        That statement was true
    
    Trial number 10, where the printout has a 59% chance of being accurate.
    Bob's printout says:
        That statement was false
    
    

We see that Alice got five false statements and five true ones, but we don't know what statements we got. Based on what he's seeing, Bob can't determine anything about the result of the experiment. Now let's go back to Alice and run some more trials.


    Trial number 11, where the printout has a 60% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    Trial number 12, where the printout has a 61% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    Trial number 13, where the printout has a 62% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 14, where the printout has a 63% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    Trial number 15, where the printout has a 64% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    Trial number 16, where the printout has a 65% chance of being accurate.
    Alice's printout says:
        The cat is dead
    
    Trial number 17, where the printout has a 66% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 18, where the printout has a 67% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 19, where the printout has a 68% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    Trial number 20, where the printout has a 69% chance of being accurate.
    Alice's printout says:
        The cat is alive
    
    

All the while Bob is in his room getting his printouts. There is still a lot of noise in her data. But she can calculate the probability again. She'll include the results she got when we were in the room with Bob and we didn't see. This time she gets:


    Alice calculates that the probability that the cat survived the experiment is 82.24%
    

OK, now Alice is feeling more confident that she knows the answer. Let's add another person, Charlie, to the experiment. Charlie doesn't get any printout, but he can talk to both Alice and Bob. They run another trial but now Alice and Bob tell their results to Charlie. Let's take a look at what Charlie sees.


    Trial number 21, where the printout has a 70% chance of being accurate.
    Charlie's printout says:
        The cat is alive
        That statement was false
    
    Trial number 22, where the printout has a 71% chance of being accurate.
    Charlie's printout says:
        The cat is dead
        That statement was true
    
    Trial number 23, where the printout has a 72% chance of being accurate.
    Charlie's printout says:
        The cat is alive
        That statement was false
    
    Trial number 24, where the printout has a 73% chance of being accurate.
    Charlie's printout says:
        The cat is dead
        That statement was true
    
    Trial number 25, where the printout has a 74% chance of being accurate.
    Charlie's printout says:
        The cat is dead
        That statement was true
    
    

For the final step of the experiment, Bob goes into the room with the experiment and opens the box. He looks for the first time to see the ultimate fate of the cat.
This concludes the experiment.

According to the definition provided by Google, the cat remains in a state of being both alive and dead until the box is opened and the cat is observed. In this experiment, there are many points where the question of whether the radioactive material has decayed could be said to be answered. The following are test points when the quantum phenomenon could be said to be resolved:

- The machine prints the 50-50 result to Alice.

- The machine prints the 51-49 result to Alice, marking the first time the experiment affected the macroscopic world.

- Alice calculates the probability the cat is alive, marking the first time a person has a good estimate of the probability.

- Bob enters the experiment and the result of the experiment could be determined by knowing what both Alice and Bob know.

- Charlie enters the experiment and become the first person to know the result of the experiment with certainty.

- Bob opens the container and obeserves the cat directly.

This thought experiment shows that none of the above explanations are satisfactory, and the only possible explanation is that the observation occurs at every point in time, when the first macroscopic entity, the Geiger counter, either detects or does not detect the radiation which causes it to release the hammer smashing the vial of poison. The presence of humans or human consciousness has no effect on quantum mechanical systems.

You can also view the [code for this post](https://nbviewer.jupyter.org/github/jss367/JupyterNotebooks/blob/master/Schr%C3%B6dinger%27s%20cat.ipynb).
