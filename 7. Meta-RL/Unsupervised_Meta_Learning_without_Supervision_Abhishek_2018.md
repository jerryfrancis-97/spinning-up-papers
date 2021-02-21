### Paper Explained

- #### Unsupervised Meta -Learning: Learning to Learn without Supervision, Abhishek et al., 2018  [Link to paper](https://arxiv.org/pdf/1806.04640.pdf)

#### **Central Idea**

This paper proposes a method to self-generate task distributions using data or environment interactions based on mutual information to support the meta-learning procedure and learn an optimal meta-learner for that environment or objective. It helps to obviate manual task designing to achieve the goal.

**Notes**

**Automating creation of tasks**

How to automate the making of good task distributions (good examples of tasks) such that the meta-task that would be learnt matches the true knowledge that needs to be learnt.

Here, designing task distributions is considered as a main bottleneck for meta-learning algorithms. Hence, the paper suggests the algorithm to self-propose task distributions by looking at the unlabeled raw data or experiencing the environment, and then meta-learn those tasks to solve the self-proposed tasks.

The tasks proposed may be random, but the task distribution isn't random, because all tasks are derived from the same data, and hence are "biased" or "constrained" to the given unlabeled data or to the behaviors of the environment. This is the additional knowledge we give to provide some context to the algorithm, OR, the thing we "pay for our lunch" according to the *No Free Lunch Theorem*.

**Deciding metrics for meta-learner**

In worst case scenario terms, an optimal meta-learner, say $$f_{*}$$ will have the maximum expected reward, across all the tasks in the distribution. We can compare the optimal meta-learner with the our current meta-learner $$f$$ using the metric of *regret*, which would use the expected reward of both $$f$$ and $$f_{*}$$ over the distribution of tasks, denoted by $$p(r_z)$$. 

​						$$REGRET(f,p(r_z)) = E_{p(r_z)} [R(f_∗,r_z)]−E_{p(r_z)} [R(f,r_z)]$$

> How to get from $$r_z$$ to $$R(f,r_z)$$ in normal terms? 
>
> $$r_z$$ or task is sampled; learning procedure $$f$$ takes task and outputs a policy $$\pi_{r_z}$$ , whose reward is calculated as $$R(\pi_{r_z})$$ or $$R(f,r_z)$$. 

As different tasks can be represented by their reward functions, we can deduce our aim as to produce an algorithm $$f$$ that can quickly learn an optimal policy $$\pi^*_{r}(a|s)$$ for a specific reward function $$r$$ or corresponding task.

Now, if the tasks are goal-specific, then the performance of the algorithm over a task having goal $$G$$ will depend on the number of times the goal $$G$$ was experienced in the training. In worst-case scenario, the task distribution can be considered as ***adversarial***, which means that any goal task that was less sampled/experienced will be shown during test-time, and due to less training on it, the regret will get maximized. This deduces that the optimal task distribution should a uniform distribution across all tasks. This idea remains the same when extended to other broader task distributions.

These conclusions about the current meta-learner/learning procedure and *"optimal and adversarial"* task distributions helps us to define our objective as a min-max objective, where our learning procedure $$f$$ tries to *minimize* the regret, and our task distribution tries to *maximize* the regret. This denoted by the following expression

​														$$min_f max_p Regret(f,p).$$ 

To calculate $$\pi$$ using $$f$$ with the constraint of being a uniform distribution $$\rho^T_{\pi}(s)$$ (probability of being at $$s$$ at timestep $$T$$ using $$\pi$$) can be challenging, especially in high-dimensional spaces. Instead, we directly calculate learning procedure $$f_{\pi}$$ without finding $$\pi$$, which is as follows.

As the tasks are proposed from a uniform distribution, and the optimal unsupervised learning procedure is the **"optimal learning procedure"** for the tasks sampled from the distribution, we can see that the main source of learning of the **"optimal learning procedure"** is actually the output/update of the optimization procedure like in meta-learning. Thus, our route of finding the **"optimal learning procedure"** is by using meta-learning over this uniform distribution.

> Hence, instead of accurately finding our desired quantity (A) to get to required quantity (B), we use an optimization method to find the ultimately-required value (B) which was dependent on our desired quantity (A), sort of skipping the *dependency* step!

**How to sample the tasks from the task distributions?**

Same as above,  instead of constructing this uniform goal distribution directly, we instead ﬁnd an optimization problem for which the solution is to sample goals uniformly.  This distribution  $$\mu(s_T | z)$$ will be conditioned in a way that the *mutual information* between the final state $$s_T$$ and the latent variable $$z$$ is maximized.

> Note : Hitting time is the first time at which a given process **"hits"** a given subset of the state space, according to stochastic process in mathematics and in our case, reaches the goal state. Given the probability of reaching the goal state $$s_g$$ at timestep $$t=T$$ using policy $$\pi$$ can be written as $$p_{\pi}^T(s_g)$$. So $$p_{\pi}^T(s_g)$$ = 0.6 means " 60 hits in 100 times" or "0.6 hits in 1 time". This implies, "no. of times for a single hit" or "time taken to hit once" is $$1/p_{\pi}^T(s_g)$$ or *Hitting time*.

Mutual information can be expressed as follows,

​														 $$I(s,z)=H(s)−H(s|z),$$ 

where $$H(s)$$ shows high entropy for state distribution (which is high for uniform distributions) and $$H(s|z)$$ shows the entropy of state distribution *conditioned* on latent variable $$z$$, which should be narrow (or consistent) to maximize $$I(s,z)$$.

Further reading  : [Blog by authors](https://blog.ml.cmu.edu/2020/05/01/unsupervised-meta-learning-learning-to-learn-without-supervision/)