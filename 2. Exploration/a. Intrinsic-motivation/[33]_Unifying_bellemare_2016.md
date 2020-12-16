### 																							Paper Explained

- #### *Unifying Count-Based Exploration and Intrinsic Motivation, Bellemare et al, 2016*	[Link to paper](https://arxiv.org/abs/1606.01868)


#### **Central Idea**

Inclusion of concept of intrinsic motivation in reinforcement learning by using exploration bonuses in the rewards across states. In count-based exploration methods, these bonuses are no. of state-visits or 'counts' by the agent. This paper introduces an algorithm to derive pseudo-counts from a density model to extend the exploration methods to the non-tabular cases in reinforcement learning.



#### 															**Notes**

Count-based exploration methods mainly depend on the number of state-visits made for each state by an agent. However, these methods may not suite well in non-tabular reinforcement learning problems, due to large amount of states present in these cases. Another reason which follows this is low state-visit counts experienced by the agent in this case as we may never get to the same state frequently.

**Solution**

 To use a density model to generalise over similar states and hence, maintain a distribution of visits of these states. Using this density model we can calculate the no. of state-visits for a state N(s), using the old and the newly updated density models, which are p(s) and p'(s) respectively and total no. of states, which is 'n' (shown in Equation 2).

**How to convert Intrinsic Motivation to Exploration bonuses?**

- Learning progress : Difference in predictive errors made by an agent after seeing a new piece of information.

- Information gain : Difference in prior and posterior distributions of a density model. Here KL-Divergence can be used to measure this difference. But computing information gain of a complex density model is intractable and hard, so a quantity called as *prediction gain* can be used. 

- Prediction gain : It is the log-difference of posterior and prior distributions of a density model.

- Other state-visit based methods: Methods like UCB, MBIE-EB, BEB can be used if no. of state-visits is known. This is  mainly used in tabular RL cases.

  From Theorem 1 and the empirical experiment of different exploration bonuses, we can show that prediction gain does well on all games, but is not the top-performing algorithm, which is (1/N)<sup>1/2</sup>.

**Experiments**

The density model chosen here is the CTS density model (Skip Context Tree Switching). Unlike other SOTA density models like pixel-recurrent neural networks, CTS' principle is count-based which helps to learn faster. This location-dependent model is used to predict the pixel's colour value conditioned on its pixel's parents, which are left-top most pixels from the current pixel (refer Figure6 : CTS filter). The probability of pixel value is then the product of the probability assigned to its pixel-parents.

**Pre-processing** : Each frame is down-sampled to 42x42, 3-bit greyscale. 

**Montezuma's Revenge**  :  Agent with exploration bonuses explores a total of 15 rooms compared to 2 rooms explored by agent with no bonuses. By  100 million frames, it achieves a score of 3349. An interesting observation here made was the use of life-loss signal. Life-loss signal is important in some games, and helps achieve high scores in them. But, in this game, life-loss signal proves to be detrimental to getting high scores.

Scenario after 200 million frames, average scores achieved are:

-  Stochastic + Life Loss: 142.50 
- Deterministic + Life Loss: 273.70 
- Stochastic without Life Loss: **1127.05** 
- Deterministic without Life Loss: 273.70

This phenomenon shows that agent knows that losing a life is better than restarting a new episode, which is analogous to human behaviour, where we also prefer losing a life to restarting with a new game.

**Exploration with Actor-Critic Methods**  :  A3C policy was regularised with an entropy cost, now becomes A3C+. The separation of policy part ( where actions happen) and critic part (where Q-values are calculated, limits the scope of exploration. In results, we see that A3C+ performs better than A3C in at least a quarter of games and also has a better median performance than A3C.

