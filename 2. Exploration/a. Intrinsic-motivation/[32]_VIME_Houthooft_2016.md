### 																							Paper Explained

- #### VIME: Variational Information Maximizing Exploration, Houthooft et al, 2016	[Link to paper](https://arxiv.org/abs/1605.09674)


#### **Central Idea**

This paper presents an exploration strategy which defines agent's curiosity as the maximization of information gain about the agent's belief on environment's dynamics. It further proposes a practical implementation using variational inference and Bayesian neural networks which efficiently handles continuous action and state spaces.

#### 															**Notes**

**Variational Inference and its role in Bayesian NNs**

If we have a probability distribution $$p$$ that is intractable, we can always look for another probability distribution $$q$$ , which is tractable (and hence can be used in calculations) and is also most close or similar to the former distribution. This would help us to infer the meaning of $$p$$ through $$q$$ . $$q$$ comes from a class of tractable distributions **Q**, so **q ∈ Q** . Hence, we can transform our inference problem into an optimization problem, which is where variational inference methods comes to the rescue.

To quantify the difference in information between $$p$$ and $$q$$ , we can use a metric called KL-Divergence and is denoted as $$KL(q||p)$$. Now this term is also intractable as it requires calculation of p. So, we here, separate p into tractable-unnormalized and intractable-normalized parts, in loose terms, as p^~^ and Z such that p=p^~^/Z . Using KL divergence between p^~^ and q ,denoted as $$J(q)$$ , helps us to derive a relation between KL(p||q) and log(Z), which helps to estimate KL(p||q) in tractable manner, hence this relation is chosen as our optimization objective.

​														$$J(q)=∑_xq(x)\log \frac{q(x)}{p(x)} $$

​																$$=∑_xq(x)\log \frac{q(x)}{p(x)}−log Z(θ)$$

​																$$= KL(q∥p)−\log Z(θ) $$  	[1]

As $$KL(q||p) \ge 0$$ (by the property of KL-Divergence), we get $$ logZ(\theta) \ge -J(q)$$ , where $$-J(q)$$ becomes our lower bound. Minimising J(q) maximises the lower bound , which in turn, results in minimising KL(q||p) , if we follow through the above equations. This lower bound is called as the *Variational Lower Bound*. Use of KL-Divergence in problems helps us to arrive at analytical or formula-based textbook solutions, which is better than approximating it over samples.

> You can read more about variational inference from [[1](https://ermongroup.github.io/cs228-notes/inference/variational/)].

In this paper, the authors are approximating the dynamic model by using a tractable distribution which is also parameterized, where this parameter $$\phi$$ represents the agent's belief. For this reason Bayesian Neural Networks (BNNs) are used as its weights maintain a distribution instead of values. Maintaining a distribution shows that BNNs can be viewed as a neural network *ensemble*. Starting with a Gaussian BNN, the variational lower bound is optimized in combination with the reparametrization trick,which is called stochastic gradient variational Bayes (SGVB) or Bayes by Backprop. Furthermore, the local reparametrization trick proposed is used, in which sampling at the weights is replaced by sampling the neuron pre-activations, which is more computationally efficient and reduces gradient variance. 

**Context of Curiosity**

In this paper, curiosity is termed as **maximising the reduction in uncertainty** about the dynamics. This means than our current actions have explored enough to more certain about the dynamics. This curiosity can be interpreted as the sum of reductions in entropy for a sequence of actions.

In information theory, the individual reduction in entropy is also called as the *mutual information* between the next state distribution and the parameter that models the dynamics model. Intuitively, it encourages the agent to take states that *maximally informative* about the dynamics model.

​								$$I (S_{t+1}; Θ|ξ_t, a_t) = \mathbb{E}_{s_{t+1}∼P(·|ξ_t,a_t)} [D_{KL}[p(θ|ξ_t, a_t, s_{t+1})||p(θ|ξ_t)]$$

​									$$ξ_t$$ : history of actions taken

​									$$Θ$$ : Dynamics model's random variable, $$ θ ∈ Θ$$

Hence, we can include this information gain (KL-Divergence between previous and new belief on the dynamics model $$p$$) as our intrinsic reward which changes are reward equation to 

​												$$r _0 (s_t, a_t, s_{t+1}) = r(s_t, a_t) + ηD_{KL}[p(θ|ξ_t, a_t, s_{t+1})||p(θ|ξ_t)]$$

where $$η ∈ \mathbb{R}+$$  is a hyper-parameter deciding the control of exploration.

We require variational inference here as the term $$p(θ|ξ_t, a_t, s_{t+1})$$ is intractable.

**Information gain as Compression improvement**

Compression improvement is defined as the improvement in reducing the information required to describe any phenomenon. In simple terms, this concept of information theory, follows the principle "the shortest, most concise description about something is the best". The amount of data used to describe is called as the description length. An agent's curiosity can also be described as the compression improvement in its description about the dynamics, and can be measured as $$C(ξ_t; \phi_{t−1}) − C(ξ_t; \phi_t)$$, where $$C(ξ; \phi)$$ is the description length of ξ using $$\phi$$ as a model.

> You can also look into [Kolmogorov Complexity](https://en.wikipedia.org/wiki/Kolmogorov_complexity#Definition) for further reading on minimum description length. 

Due to the usage of variational inference,we can view the description length as the negative of the variational lower bound. Hence, we can write the compression improvement in this situation as  $$L[q(θ; \phi_t), ξ_t] − L[q(θ; \phi_{t−1}), ξ_t]$$, where the variational lower bound is as follows: 

​												$$L[q(θ; \phi), D] = \mathbb{E}_{θ∼q(·;\phi)} [\log p(D|θ)] − D_{KL}[q(θ; \phi)||p(θ)]$$

The authors show that if we assume that our approximator accurately defines the dynamics model, i.e., $$q(\theta;\phi) = p(θ|ξ_t)$$ , then this new improvement reduces to $$D_{KL}[p(θ|ξ_{t−1})||p(θ|ξ_t)]$$ , which is the reversed KL-Divergence. This variant can also be used for curiosity other than information gain. The paper states that both the variants are not that different.

 **Experiments**

VIME was experimented on different games like CartPole, MountainCar, DoublePendulum, CartPoleSwingup, in locomotion tasks Walker2D and HalfCheetah and one difficult hierarchical task SwimmerGather, and following results were achieved:

- VIME performed well in systems having both sparsely shaped and well-shaped reward structures. Here, average return excluding intrinsic rewards were considered for evaluation.
- It improved the exploration when combined with algorithms like REINFORCE, ERWR, which suffer from poor exploration due to their premature convergence to sub-optimal policies.
- Showed good results in difficult game SwimmerGather.
- Trade-off between exploration and exploitation using $$\eta$$ : Higher $$\eta$$ showed higher curiosity leading to more exploration. Whereas, very low $$\eta$$ values should reduce VIME to traditional Gaussian control noise. VIME explores in efficient-diffused pattern over the state-space.

