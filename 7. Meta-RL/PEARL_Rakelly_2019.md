### 																							Paper Explained

- #### Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables, Rakelly et al, 2019. 	[Link to paper](https://arxiv.org/pdf/1903.08254.pdf)

#### **Central Idea**

This paper introduces an off-policy meta-RL method that disentangles task inference and control. It uses a probabilistic encoder to estimate task belief, which also enables the posterior sampling for structured-extended exploration. Its integration with off-policy RL method achieves both meta-training and adaptation efficiency.

#### 															**Notes**

Meta-RL is sample-inefficient when  it comes to meta-training part, where it requires on-policy data for meta-training and adaptation processes. To make it sample-efficient, off-policy data can be used as an experience, but the problem is that meta-learning operates on the principle "meta training time should match meta testing time", or, "if we are meta-learning over $$n$$ tasks, training should have sets of $$n$$-tasks' examples". This is not possible with off-policy data, as it may present experience over tasks that may be different from the new task the agent may explore in meta-testing.

To incorporate off-policy, the authors viewed the adaptation of task in meta-learning as a POMDP (Partially observed MDP), where the hidden state is the specific task or its MDP. As agent is unaware of the task it currently in, a belief over the task can be used to describe this uncertainty. This can be done using a latent representation over prior experience, like an encoder.

**How to learn latent contexts?**

Suppose that we have a prior experience in the form of history of past transitions, which we can call *'context'* , **$$c$$**. Here, $$c ^T_n = (s_n, a_n, r_n, s'_n )$$ be one transition in the task T so that $$c^T_{1:N}$$ comprises the experience collected so far. To encode the context into a latent form $$Z$$  the concept of variational inference is used, by training an inference network $$q_{\phi}(z|c)$$, parameterized by $$\phi$$, that estimates the posterior $$p(z|c)$$. 

The inference network is optimized to model state-action value functions or to maximize the returns from a policy over a distribution of tasks. Using log-likelihood as objective, the *variational lower bound* is as follows,

​										$$E_T [E_{z∼q_{\phi}(z|c^T )} [R(T , z) + βD_{KL}(q_{\phi}(z|c^T )||p(z))]]$$.   

$$p(Z)$$ is a gaussian prior over $$Z$$, $$R(T,z)$$ can be taken as return or state-action values (can be taken as other objectives too). The KL-Divergence term shows the restriction/bottleneck of z to the information only provided by the context $$c$$, also as an example of mutual information between $$Z$$ and $$c$$. This bottleneck mitigates overfitting in training tasks.

**Designing $$q_{\phi}(z|c)$$**

As the transitions follow *Markov* property, the ordering of $$(s,a,s',r)$$ tuples is not required to infer the task-belief. Hence, a permutation-invariant encoder can be used, where  $$q_{\phi}(z|c_{1:N} )$$ is calculated as a product of independent factors 

​														$$q_{\phi}(z|c_{1:N} ) ∝ Π^N_{n=1}Ψ_{\phi}(z|c_n)$$. 

To keep the method tractable, we use Gaussian factors $$Ψ_{\phi}(z|c_n) = N (f ^µ _\phi (c_n), f^σ _\phi (c_n))$$, which result in a Gaussian posterior. Here $$f_\phi$$ is a neural network which outputs $$\mu$$ and $$\sigma$$ as mean and standard deviation of $$c_n$$.

**Posterior sampling for exploration**

Also known as Thompson Sampling, it works in the following way : assume the information you have is true, do the best/optimal using it, and finally, use the new experience acquired to update your knowledge about the information you earlier had. This ultimately, narrows your knowledge to the best or most accurate meaning. 

Similarly, here,  during training time, the prior over $$z$$ is learnt to represent task-distribution. Then, during the meta-test time, a $$z$$ (hypothesis) is sampled from the prior, the model takes actions using it in environment, and updates the posterior with the new experience learnt. By further collecting more new experiences, the model can make a better guess about the current task, as the posterior narrows it down.

**De-coupling data streams for inference and RL policy learning**

> *"Data to train the encoder need not be the same as the data used for training policy"*

Here the latent context $$z$$ can be considered as part of the state in the main off-policy RL loop, while the uncertainty by encoder $$q_{\phi}(z|c)$$ provides the structured yet stochastic exploration. This is implemented by using a *Context Sampler* $$S_c$$ to sample context batches for the encoder. These batches are taken from the most recently collected batch of data, recollected every 1000 meta-training optimization steps. This is not a strict on-policy, but good enough to retain on-policy performance from recently-collected data. The off-policy RL method uses batches drawn uniformly from the entire buffer.

**Implementation & Experimental results**

SAC is an off-policy actor-critic method which is based on maximum entropy RL objective. The inference network is trained using the gradients from the critic, which models the network as a distribution over different Q functions.

- PEARL uses 20-100x fewer samples during meta-training than previous meta-RL approaches while improving final asymptotic performance by 50-100% in five of the six domains.
- Replacing encoder with RNN with de-correlated transitions shows comparable performance at the cost of slow optimization, but low performance when using RNN with whole trajectories.
- Sampling context off-policy hurts performance, but using same data batches for RL and context helps in performance due to correlation.
- Choosing a deterministic encoder heaves the exploration contribution to the RL policy learning, which drops the performance.



Further reading : [Blogpost in BAIR ](https://bair.berkeley.edu/blog/2019/06/10/pearl/) by *Kate Rakelly* 