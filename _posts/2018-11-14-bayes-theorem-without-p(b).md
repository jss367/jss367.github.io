---
layout: post
title: "Bayes' Theorem without P(B)"
description: "This post is about Bayes' Theorem and how to use them."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/hoary-headed-grebe.jpg"
tags: [Statistics]
---

Bayes' Theorem is a fundamental concept in probability theory and statistics, offering a powerful framework for updating our beliefs in light of new evidence. This theorem is particularly useful in fields like machine learning, medical diagnostics, and even everyday decision-making. But what happens when we lack certain probabilities, specifically P(B)? This post talks about how Bayes' Theorem works and explores strategies to apply it even without direct knowledge of P(B).

# Background

Bayes' Theorem relates the conditional and marginal probabilities of random events. It's mathematically expressed as:

$$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$


Where:

* $$ P(A|B) $$ is the probability of event A occurring given that B is true.
* $$ P(B∣A) $$ is the probability of event B occurring given that A is true.
* $$ P(A) $$ is the probability of A occurring. This is know as the "prior".
* $$ P(B) $$ is the probability of B occurring.

A and B are variables that can represent anything, but, in general, A is the the hypothesis you want to test and B is the new evidence. You want to know what is the probability of hypothesis A given that you just saw evidence B.

I often think of it like this:

$$ P(A|B) = P(A) \times \frac{P(B|A)}{P(B)} $$

because P(A) is what I previously believed to be true, and $$ \frac{P(B|A)}{P(B)} $$ is the ratio of the probability that I would see some evidence given A over the probability of seeing that evidence in general.

# Missing P(B)

Often, the direct probability of evidence B, P(B), is unknown or difficult to determine. Fortunately, you can break down P(B) using the law of total probability. This law states that the probability of an event can be expressed as the sum of the probabilities of the event occurring in conjunction with mutually exclusive and exhaustive events. This includes the probability of B happening when A is true and when A is not true.

$$ P(B) = P(B|A)P(A) + P(B|\neg A)P(\neg A) $$

We can then plug that into Bayes' Theorem to get:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}= \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)} $$

By decomposing P(B) using the law of total probability, we can still harness the power of Bayesian inference to update our beliefs in light of new information.
