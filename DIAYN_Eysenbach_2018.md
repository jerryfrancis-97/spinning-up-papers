### Paper Explained

- #### Diversity is all you Need, DIAYN by Eysenbach et al., 2018 [Link to paper](https://arxiv.org/pdf/1802.06070v6.pdf)

#### **Central Idea**

This paper proposes a method to learn new, useful and diverse skills without requiring supervision, that is, without a reward function. This method maximizes an information theoretic objective -- which is used to discriminate different learned skills, while using a maximum entropy policy.

**Notes**

**Major contributions**

- It proposes a method for learning useful skills without any rewards
- It shows that this simple exploration objective results in the unsupervised emergence of diverse skills, such as running and jumping, on several simulated robotic tasks
- It proposes a simple method for using learned skills for hierarchical RL and find this methods solves challenging tasks
- It demonstrates how skills discovered can be quickly adapted to solve a new task
- It shows how skills discovered can be used for imitation learning



**Distinction from related works**

- Learning skills without reward function circumvents the need of reward design, makes the method task-agnostic.
- Maximum entropy policies help to produce a diverse set of skills, where the skills achieve maximum entropy in aggregate
- Prior distribution is updated or fixed, rather than learnt -- as in Variational Intrinsic Control (VIC), which reduces the number of skills sampled as the training progresses.
- The discriminator looks at every state, rather than the final state ,as in VIC, which gives additional reward signal
- It proposes an objective for learning many policies instead of a single policy, prevalent in the prior works which use an intrinsic motivation objective. 



**Role of information theory**

A visible representation of a skill comes by observing some aspect of the path or trajectory. Here, we use the states of the trajectory to understand the skill executed by the agent, which also means that skill decides the states an agent visits. So, we maximize the mutual information between skill $$Z$$ and states $$S$$ , denoted by $$I(S;Z)$$. 

We distinguish the skills using the skills, hence we minimize the mutual information  between skills and actions given the state, which is $$I([A;Z]/S)$$. We also maximize the entropy of the mixture of policies, denoted by $$H(A/S)$$.

Combining the above terms, we maximize our final objective as,

​															$$ F(θ) = I(S;Z) + H[A | S] − I(A;Z | S) $$

​															         $$= H[Z] − H[Z | S] + H[A | S, Z]$$ 

$$H[A | S, Z]$$ means that each  skill should act as randomly as possible.

Approximating $$p(z|s)$$ with $$q_\phi(z|s)$$ , we introduce variational lower bound on our objective $$F(\theta)$$, denoted by $$ G(θ, φ)$$. Hence,

​					     		$$F(\theta) ≥ H[A | S, Z] + E_{z∼p(z),s∼π(z)} [\log q_φ(z | s) − \log p(z)] \equiv G(θ, φ)$$

We maximize the expectation in G by replacing the task reward with the following pseudo-reward: 

​																$$r_z(s, a) = \log q_φ(z | s) − \log p(z)$$



**Experiment and results**

DIAYN is implemented with soft actor-critic , where the term $$H[A | S, Z]$$ is regularized with $$\alpha$$ . The value $$\alpha = 0.1$$  provides a good-tradeoff between exploration and discriminability. During unsupervised learning, we sample a skill $$z ∼ p(z)$$ at the start of each episode, and act according to that skill throughout the episode. The agent is rewarded for visiting states that are easy to discriminate, while the discriminator is updated to better infer the skill $$z$$ from states visited. Entropy regularization occurs as part of the SAC update.

***Differences with VIC on diversity of skills***

VIC samples more diverse skills in a more frequent manner, and hence only those skills will get training signal to improve. This causes **Matthew Effect**. A summarized definition of Matthew effect of accumulated advantage is seen in the adage "the rich get richer and the poor get poorer".

***Does discriminating on single states restrict DIAYN to learn skills that visit disjoint sets of states?***

No, as seen in 2D navigation experiment, the skills discriminate themselves only after exiting the hallway. This is because the point agent is incentivized to maximize the cumulative reward, and therefore, it sometimes takes actions that give no reward initially to reach the high-rewarding states.

***How can DIAYN leverage prior knowledge about what skills will be useful?***

By changing the discriminator approximator from $$E[\log q_\phi(z|s)]$$ to $$E[\log q_\phi(z|f(s))]$$ , where $$f(s)$$ is a function of the observations. For example, in the ant navigation task, $$f(s)$$ could compute the agent’s center of mass, and DIAYN would learn skills that correspond to changing the center of mass.

***IMITATING AN EXPERT : . Can we use learned skills to imitate an expert?***

Imitation learning replaces the existing policy with a similar yet differentiable policy, which might be easier to update in response to new constraints or objectives.  Given the expert trajectory, we use our learned discriminator to estimate which skill was most likely to have generated the trajectory. This optimization problem, which we solve for categorical z by enumeration, is equivalent to an M-projection.

***What is M-projection?***

In the imitation learning task, each skill visits some distribution over states, and the expert also visits some distribution over states. We do the most straightforward approach: we compute the distance between each skill and the expert, and taking the closest skill. The slightly tricky part is computing distance between distributions over states. If we use the KL-Divergence as our distance metric, then our approach is called an M-Projection. [This article](http://www.ams.org/journals/notices/201803/rnoti-p321.pdf) has more details on M-Projections and I-Projections. (referred from [here](https://github.com/flrngel/understanding-ai/issues/7))



**Use of $$\log p(z)$$ in reward**

If the agent interacts in an infinite horizon case, it will eventually end at a terminal state. At this state, it will be difficult to discriminate the skills, so its estimate is $$\log q(z | s) = \log(1/N)$$. As we work in finite length episodes, this term is used to show the *artificial termination* of episode assuming that the agent ran infinitely many episodes to reach the terminal state. Also $$q(z|s) >=p(z)$$ ,which makes their difference positive, and provides optimism to agent to continue. **In some environments, such as mountain car, it is desirable for the agent to end the episode as quickly as possible. For these types of environments, the log p(z) baseline can be removed**