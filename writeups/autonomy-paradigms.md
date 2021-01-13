# Autonomy paradigms

## Grids

This is an exact concretization of the N choose K model. On each trial, we present participants with a gamble, displayed as a grid. Each cell contains a reward that is initially occluded but can be revealed by clicking. After exhausting a budget of $s$ clicks, the participant decides whether they want to accept the gamble or not. If they accept, we randomly choose $k$ cells (according to some non-uniform distribution) and the participant chooses among them.

The cells in the grid are approximately sorted along two dimensions, selection probability and reward. Thus, very probable and high value cells would be in, e.g., the bottom right. This allows us to measure the extent to which participants are sampling according to probability (lower vs upper) vs. value (right vs left).

The problem with this paradigm is that it's super artificial. This motivates...

## Storage wars

In this paradigm (_not affiliated with [the A&E show](https://en.wikipedia.org/wiki/Storage_Wars) of the same name_), participants are presented with a storage unit containing several boxes, and they have the opportunity to purchase an item from the unit. However, they cannot choose exactly which item they get. Instead, a random selection of $k$ items (each of which has some associated resale value) are pulled from the unit and the participant can take the most valuable one. Before making their decision, the participant can check the contents of $s$ boxes.

There are three types of boxes: cardboard boxes tend to have low-value items, plastic bins have medium-value items, and wooden chests have high-value items. Participants are instructed that the employee who picks which boxes to pull (the contents of which the participant chooses between) is ambivalent between the different box types; however, she is somewhat lazy and thus tends to pick boxes that are close to the entrance of the unit.

For low $k$, participants should primarily sample the boxes close to the entrance. For high $k$, participants should primarily sample the wooden chests (with high expected value).

## Naturalistic decisions

Fiery has proposed two naturalistic decisions that get at this question (see fiery-paradigms.pdf). In the first paradigm, we ask participants to imagine that their friend is buying them dinner and they get to choose the restaurant. In the high control case, they also choose their dish; in the low control case, the friend randomly selects an item on the menu. We then ask them to self-report what dishes they considered (and I suppose also how much they like those dishes).

The second paradigm is similar, but uses states and cities in place of restaurants and dishes. One issue here is that in the high control case, I think people will not really do any state evaluation and will instead just pick a city to live in. I imagine many people already know what large city they would most like to live in, so there might not be any decision-making process at all.

I don't think these are good choices for the initial experiment due to lack of experimental control. However, I think something like these could be good to include as an additional experiment to demonstrate external validity.

## Words and letters

This task is described in more detail in fiery-paradigms.pdf.

In this task the two stages of choice involve first picking a word from a set and then choosing a letter from that word. Each letter is associated with a reward (e.g. A=1¢, B=2¢, ...). After participants choose a word, we randomly select $k$ letters[^ Fiery proposed a binary control manipulation where they either get a random letter or they get their choice out of all the letters] and the participant selects among them.

In this case, the sampling is not observable. Instead, we will employ self-report, asking participants what they considered. This approach has worked well when only asking participants to list which words they considered. However, it is not clear that they will also be able to report which _letters_ they considered in each word. Asking them to create an exhaustive list is clearly impractical. We could potentially ask them which letters they considered in their chosen word (and/or a randomly chosen word from the set). However, I'm skeptical that people will be able to report this level of detail in their decision making process.

One advantage of this paradigm is that it can be most naturally extended to ask questions about what comes to mind outside of the online decision making and about normality judgments.


## Pachinko

I originally came up with this paradigm when I was unable to create a naturalistic two-step problem. I don't think it's the right first experiment, but it would allow us to ask a richer set of questions about planning (rather than one-step sampling).

The paradigm is a modification of the classic Pachinko game.[^ It might actually closer to Plinko or the "bean machine"] There are a bunch of pins in a board and we drop a ball at a fixed location on the top. The ball bounces down and eventually lands in one of several bins at the bottom of the board . Each bin has some value, and these are (approximately) sorted such that the bins on the right have higher value. If the pins are arranged in a grid, the dynamics [induce a binomial distribution](https://arxiv.org/abs/1601.05706) on the outcomes such that the ball is more likely to land in the middle bins.

![](https://i.stack.imgur.com/M7j2V.png)

The critical modification is that some pins have a switch so that the participant can decide which direction the ball goes if it hits that pin. The proportion of switch pins corresponds to the degree of control.

In the simplest version of the task, we pose an accept/reject decision as in the grid task. Participants can first check the value of any number of bins (with a time cost) and then they decide if they want to play or not. We are interested to see to what extent people prioritize probability (checking bins in the middle) vs value (checking bins on the right side) as a function of the degree of control (proportion of switch pins).

A key difference between this task and the previous ones is that it involves a true sequential decision problem.[^ The previous tasks can be construed as two-step decisions, but it feels just as---if not more---natural to think of it as a one-step decision with a different reward function, especially in the extreme cases of zero vs. full control (mean vs. max of the outcome values)] 
This allows us to ask a lot of new questions. Three of particular interest:

1. How does controllability affect the amount of planning people do? Falk, Noah, and Quentin have [some early work](https://www.researchgate.net/profile/Falk_Lieder/publication/258919326_LiederF_and_Goodman_ND_and_Huys_QJM_2013_Controllability_and_resource-rational_planning_Cosyne_Abstracts_2013_Salt_Lake_City_USA/links/5455912f0cf2bccc490cce17/Lieder-F-and-Goodman-ND-and-Huys-QJM-2013-Controllability-and-resource-rational-planning-Cosyne-Abstracts-2013-Salt-Lake-City-USA.pdf) on this, but I don't think they've followed up on it. Their basic result is that more controllability should lead to more planning, but I think there will be more nuance.
2. Does propensity to plan play a role in the link between learned helplessness and depression? In particular, if a person infers that she has low control, then she will rationally not engage in planning. This could create a vicious cycle because not planning will lead to not achieving desired outcomes which will reinforce the belief that she doesn't have control. I'm not sure to what extent this is interestingly different than the standard learned helplessness story.
3. Under what circumstances is outcome-sampling a resource-rational alternative to full planning (something like MCTS)? I think we'll find that outcome sampling is preferable in both the high and low control cases: when you have no control, there's nothing for you to plan; when you have complete control, you can just assume you'll get the best outcome without worrying about how exactly you'll get there. 

A note on implementation: I can create an artificial version of this with fairly minimal modifications to my Mouselab MDP paradigm ([demo](https://webofcash2.herokuapp.com/?exp=1&variance=constant&show=task&length=full&skip=8&clickDelay=0)). If results look promising there, I think it would be worth implementing a more engaging version that actually looks like Pachinko, perhaps with the help of a CS undergrad (or maybe even a contractor).




<!-- For example, we can compare the outcome-sampling strategy to a planning strategy (e.g. MCTS). Previous work  suggests that the value of planning increases with controllability.
    We could probably confirm this prediction based on reaction times alone, but we could also introduce more process tracing, letting participants click to check whether a pin has a switch. We could also dissociate controllability from stochasticity by introducing pins that deterministically send the ball left or right. -->

<!-- By introducing additional mechanisms such as locked switches (that deterministically send the ball left or right) and process tracing on the pins (click to check what kind of pin it is), we can empirically measure things how control influences the degree of planning vs outcome-sampling and forward vs. backward  -->
