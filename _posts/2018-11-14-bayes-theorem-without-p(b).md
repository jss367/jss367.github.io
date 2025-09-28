---
layout: post
title: "Bayes' Theorem without P(B)"
description: "This post is about Bayes' Theorem and how to use them."
feature-img: "assets/img/rainbow.jpg"
thumbnail: "assets/img/hoary-headed-grebe.jpg"
tags: [Bayesian Statistics, Statistics]
---

Bayes' Theorem is a fundamental concept in probability theory and statistics, offering a powerful framework for updating our beliefs in light of new evidence. This theorem is particularly useful in fields like machine learning, medical diagnostics, and even everyday decision-making. But what happens when we lack certain probabilities, specifically P(B)? This post talks about how Bayes' Theorem works and explores strategies to apply it even without direct knowledge of P(B).

## Background

Bayes' Theorem relates the conditional and marginal probabilities of random events. It's mathematically expressed as:

$$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$

Bayes factor

$$ \frac{P(B|A)}{P(B|\neg A)} $$

Where:

$$ P(A∣B) $$ is the probability of event A occurring given that B is true.

$$ P(B∣A) $$ is the probability of event B occurring given that A is true.

$$ P(A) $$ is the probability of A occurring. This is known as the "prior".

$$ P(B) $$ is the probability of B occurring.

A and B are variables that can represent anything, but, in general, A is the hypothesis you want to test and B is the new evidence. You want to know what is the probability of hypothesis A given that you just saw evidence B. When it's used in that context, you'll often see it written like this:

$$ P(H|E) = \frac{P(E|H) \times P(H)}{P(E)} $$

I often think of it like this:

$$ P(A|B) = P(A) \times \frac{P(B|A)}{P(B)} $$

because P(A) is what I previously believed to be true, and 

$$ \frac{P(B|A)}{P(B)} $$

is the ratio of the probability that I would see some evidence given A over the probability of seeing that evidence in general.

## Missing P(B)

Often, the direct probability of evidence B, P(B), is unknown or difficult to determine. Fortunately, you can break down P(B) using the law of total probability. This law states that the probability of an event can be expressed as the sum of the probabilities of the event occurring in conjunction with mutually exclusive and exhaustive events. This includes the probability of B happening when A is true and when A is not true.

$$ P(B) = P(B|A)P(A) + P(B|\neg A)P(\neg A) $$

We can then plug that into Bayes' Theorem to get:

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)}= \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)} $$

By decomposing P(B) using the law of total probability, we can still harness the power of Bayesian inference to update our beliefs in light of new information.

## Odds Form of Bayes' Theorem

Another powerful way to express Bayes’ Theorem is in terms of **odds** instead of raw probabilities. Odds are just another way of describing how likely something is compared to its alternative. If the probability of an event is $$p$$, then the odds are:

$$ \text{odds}(A) = \frac{p}{1 - p} $$

So if \( P(A) = 0.2 \), the odds are \( 0.2 / 0.8 = 0.25 \), which you can read as “one to four against.”


## From Bayes’ Rule to Odds

Start with Bayes’ Theorem for both $$A$$ and $$\neg A$$:

$$
P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)}
$$

$$
P(\neg A|B) = \frac{P(B|\neg A)P(\neg A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)}
$$


Bayes factor

$$ \frac{P(B|A)}{P(B|\neg A)} $$

Now form the ratio:

$$
\frac{P(A|B)}{P(\neg A|B)} =
\frac{\tfrac{P(B|A)P(A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)}}
     {\tfrac{P(B|\neg A)P(\neg A)}{P(B|A)P(A) + P(B|\neg A)P(\neg A)}}.
$$

The denominators cancel, leaving:

$$
\frac{P(A|B)}{P(\neg A|B)} =
\frac{P(B|A)P(A)}{P(B|\neg A)P(\neg A)}.
$$

Finally, separate into two ratios:

$$
\frac{P(A|B)}{P(\neg A|B)} =
\frac{P(A)}{P(\neg A)} \times \frac{P(B|A)}{P(B|\neg A)}.
$$

Bayes factor

$$ \frac{P(B|A)}{P(B|\neg A)} $$

Notice the ratios that appear:

Bayes factor

$$ \frac{P(B|A)}{P(B|\neg A)} $$

- The prior odds: $$ \frac{P(A)}{P(\neg A)} $$

Bayes factor

$$ \frac{P(B|A)}{P(B|\neg A)} $$

- The Bayes factor (how strongly the evidence favors A over ¬A): $$ \frac{P(B|A)}{P(B|\neg A)} $$
- The Bayes factor (how strongly the evidence favors A over ¬A): $$ \frac{P(B|A)}{P(B|\neg A)} $$
- The posterior odds: $$\dfrac{P(A \mid B)}{P(\neg A \mid B)}$$

Bayes factor

$$ \frac{P(B|A)}{P(B|\neg A)} $$

The Bayes factor (how strongly the evidence favors A over ¬A): $$ \frac{P(B|A)}{P(B|\neg A)} $$

The Bayes factor: $$ \frac{P(B|A)}{P(B|\neg A)} $$

Test: $$ \frac{P(A)}{P(\neg A)} $$

Test2: $$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$

Test2: $$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B)} $$

Test2: $$ P(A|B) = \frac{P(B|A) \times P(A)}{P(B|\neg A)} $$

So now, we can translate this to words:

$$ \text{Posterior odds} = \text{Prior odds} \times \text{Bayes factor} $$

## Why This Matters

This form is elegant because:

- You don’t need to compute messy denominators.
- Updating is multiplicative: every new piece of evidence just multiplies your odds by another Bayes factor.
- It’s easier to see the “weight of evidence” — a Bayes factor greater than 1 shifts belief toward A, while less than 1 shifts it away.

For practical Bayesian reasoning, especially when evidence comes in sequentially, thinking in terms of odds and Bayes factors often feels much more natural.


