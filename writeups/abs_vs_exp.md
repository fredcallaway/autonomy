---
title: Absolute vs exponentiated value
date: January 13
---

## Setup
(same as before)

Suppose you are given a basket containing $n$ balls with normally distributed values, $u_i$.

You can accept or reject the basket. If you reject, you get nothing. If you accept, I randomly draw $k$ balls (without replacement) and you get the best one.

Before making a choice, you get to draw $s$ balls from the basket. One reasonable thing to do is to look at the mean of those samples and accept the basket if the mean is greater than zero. Assuming you use this overall strategy, how should you decide which balls to sample?

## Model
We assume that the samples are drawn with probability proportional to their probability of occuring, multiplitied by a convex combination of exponentiated value and absolute value:

$$
f(i) \propto p_i \cdot \left((1 - β) |u_i| +\ β e^{u_i}\right)
$$

After sampling $s$ outcomes, we reweight them to account for the sampling bias (as in importance sampling). To capture a range of strategies between simple biased sampling and full reweighting, we compute our "expected value" as

$$
\hat{u} = \frac{1}{s} \sum_{j\in\mathcal{S}} 
    \left(\frac{p_j}{f(j)}\right)^α \cdot u_j 
$$
where $\mathcal{S}$ is the set of outcomes we sampled and $α$ is a parameter that controls the degree of reweighting: $α = 1$ gives importance sampling, $α=0$ gives no reweighting, and intermediate values are, well, intermediate.

## Results

Rather than building up complexity I'm going to throw everything at you at once. Note that these results assume uniform outcome probability, but they are qualitatively similar if we let $p_i$ vary.

![Reward attained as a function of the degree to which high vs. extreme values are sampled, $β$, and the degree to which the sampling bias is accounted for by reweighting, $α$. The panels vary in the amount of control, $k$, and the number of samples that are taken before the decision is made, $s$.](../model/figs/abs_exp_importance_equalprob_grids.png)

Several observations:

- For $s=1$, reweighting doesn't matter. This makes sense because all that matters is the decision is made based on the sign of the sample.
- In no case does reweighting improve performance (for the best $β$). It's surprising to me that reweighting is actually harmful in the $k=1$ case. I'm not sure how robust this will be to changes in the problem setup, in particular the underlying value distribution (Gaussian). Note that the goal of importance sampling is usually to minimize error in your estimate, whereas in our case you just need to get the sign right to make the correct choice.

Given that importance sampling appears to be unnecessary in our case, let's fix $α=0$.

![Same as above with $α=0$.](../model/figs/abs_exp_equalprob.png)

- We see the basic prediction that sampling high-value outcomes is better as control increases. For low values of $k$ we see a compromise between high and extreme value, which is nice.
- More samples (higher $s$) also encourages more sampling of high vs. extreme outcomes. This could be an interesting thing to manipulate (time pressure). On the other hand, it could be a nuisance variable because when $s$ is high, the transition from extreme to high value sampling is more sharp.


















