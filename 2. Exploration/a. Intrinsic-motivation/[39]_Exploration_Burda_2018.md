### 																							Paper Explained

- #### *Exploration by Random Network Distillation, Burda et al, 2018*	[Link to paper](https://arxiv.org/abs/1810.12894)


#### **Central Idea**

Use of Random Network Distillation as an exploration bonus along with the flexibility of combining intrinsic and extrinsic rewards to significant progress in the exploration tasks. A fixed randomly initialised network generates an error for observation,the difference of which is considered as the "novelty" of observation. 

#### 															**Notes**

Rewards with exploration can be defined as r~t~ = e~t~ + i~t~ , where e~t~ is the extrinsic reward (usual reward) and i~t~ is the intrinsic reward (exploration purposes). For i~t~ , many methods like state-visit methods (UCB, etc.), count-based methods, and forward dynamics methods have been used. Other prediction methods involves knowing about special information of an environment can prove to be useful if available.

As the error in such prediction methods decreases as the agent experiences more similar states, any simple prediction problem like predicting a constant zero function can also work as exploration bonuses.

**Intuition about Random Network Distillation**

Imagine a model trained on dataset that classifies images of dogs and cats. Here after training, if the model is shown a dog's image which was completely unseen by the model, the MSE or the model error will be huge. This error may decrease after showing some more unseen examples of the dog's image. This implies that model outputs a high error to examples which it can relate to or hasn't seen in the past, and thus, unseen examples are "new"  or "novel" for the model. Hence we can relate such errors as a measure of "novelty" or exploration bonuses.

RND takes a fixed random network *f* and uses a predictor network *f\** that tries to mimic the random network's weights by training on data collected by the agent. The objective is to minimise MSE between them || *f*(x) - *f*\*(x)||^2^  using parameters of f\* with the help of gradient descent. A novel state will give high MSE as output. A caveat to this is that if a strong optimisation algorithm is used here, it will be able to mimic the random network perfectly and we will not be able to use its error as bonus. 

 **Noisy TV problem**

It is a problem in which agent gets trapped by its own curiosity about the environment. It is similar to the situation where a player gets trapped into playing a game of luck just when the player starts to get different and new outcomes. Here, the authors actually placed a noisy TV (which shows different images of channels) in the environment, and agent was into looking into it, as its randomness was generating new state, which in turn was giving exploration rewards to agents.

*Sticky actions* : A noisy controller held by the agent takes the last action instead of current action with some probability. This method is used to train agents in deterministic games in Atari so that the agent doesn't follow determinism and avoids memorising a good path while training. These actions make changing of rooms unpredictable.

If our exploration methods are based on next state prediction, we encounter the noisy TV problem. RND obviates this noisy TV problem as the target network can be made deterministic.   

**Deciding between non-episodic and episodic returns**

Keeping intrinsic rewards non-episodic helps in better exploration of the environment. Here, return calculation is not stopped at "game-over" state. As intrinsic rewards are meant only for exploration, number of episodes to reach a novel state is unimportant if we mange to reach a novel state. Using episodic returns will make the agent think about restarting the whole game from beginning, making it wary of "game-over" states (risk-averse behaviour). A strategy that finds a reward close to the beginning of the game can be used to deliberately restart a game, making an endless cycle. This strategy can used with non-episodic returns for extrinsic rewards. 

**Single versus Dual value-heads**

Total reward can be decomposed as a sum R = R~E~ + R~I~ , where R~E~ and R~I~ are the extrinsic and intrinsic returns respectively. Hence we can use two value functions V~E~ and V~I~ separately for their respective returns, and combine them to give the total value function V = V~E~ + V~I~ . This dual value heads can be useful for exploration because the extrinsic reward function is stationary (doesn't change),  whereas the intrinsic reward function are non-stationary (can change).

Reward and Observation Normalisation

Reward Normalisation is done on intrinsic rewards by dividing it by a running estimate of the standard deviations of the intrinsic returns. Scale of these rewards can vary greatly, making it difficult for hyper-parameter tuning. 

Observation Normalisation is done by subtracting the running mean and dividing by the running standard deviation. This is also clipped between -5 and 5.

**Experiments**

 Most experiments are run for 30K rollouts of length 128 per environment with 128 parallel environments, for a total of 1.97 billion frames of experience. Two policy architectures were used here, RNN and CNN, where 

- In the experiment between non-episodic and episodic returns where extrinsic rewards were avoided (sole exploration purpose), the agent performed better in non-episodic setting than in the episodic setting. Extrinsic rewards cannot be fully excluded as some rewards for example, getting key or other objects in the game allow the agents to explore other rooms. However, the agent isn't greatly influenced by these extrinsic rewards. 
- Agent in the non-episodic setting with γ~I~ = 0.999 explores more rooms than γ~I~ = 0.99, with one of the runs exploring 21 rooms (here, γ~I~ is the discount factor for intrinsic rewards). An *interesting fact* is that even without considering the extrinsic rewards, RND method managed to explore more than *half of the rooms* in Montezuma's Revenge game.
- Combining non-episodic stream of intrinsic rewards with the episodic stream of extrinsic rewards outperforms combining episodic versions of both steams in terms of number of explored rooms, but performs similarly in terms of mean return. 
- Single value estimate of the combined stream of episodic returns performs a little better than the dual value estimate. The differences are more pronounced with RNN policies. CNN runs are more stable than the RNN counterparts. 
- In the experiment of comparing different discount factors, the performance of the RND agent with γ~E~ ∈ {0.99, 0.999} and γ~I~ = 0.99 is compared. It was seen that increasing γ~E~ to 0.999 while holding γ~I~ at 0.99 greatly improves performance. It also seen that further increasing γ~I~ to 0.999 hurts performance.
- It was also seen that using parallel environments for training agents gives larger batches of experience for the  same amount of updates. This improvement saturates earlier in CNN architecture than in RNN.
- RND method works better than PPO without an exploration bonus and PPO with an exploration bonus based on forward dynamics error, which were the baselines used for this comparison.

**Exploration in Montezuma's Revenge**

RND method works well in local exploration which tries to avoid immediate, short-term rewarding decisions like whether to take an object or not. But long term exploration is beyond the reach of this method. 

To solve the first level of Montezuma’s Revenge, the agent must enter a room locked behind two doors. There are four keys and six doors spread throughout the level. Any of the four keys can open any of the six doors, but are consumed in the process. To open the final two doors the agent must therefore forego opening two of the doors that are easier to find and that would immediately reward it for opening them.

To forego such a rewarding decision, the agent must get enough intrinsic reward. The RND method doesn't actually give enough incentive to try this strategy, but rarely stumbles on this strategy as said in the paper.

