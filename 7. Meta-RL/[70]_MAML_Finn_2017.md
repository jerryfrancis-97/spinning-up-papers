### 																							Paper Explained

- #### Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks, Finn et al, 2017. 	[Link to paper](https://arxiv.org/abs/1703.03400)

#### **Central Idea**

This paper proposes a meta-learning algorithm that is model-agnostic, ie. it is compatible with any model trained with gradient descent, and requires a small number of gradient steps and little amount of training data to learn a variety of tasks in different learning problem domains like supervised learning like regression, classification and RL, and achieve SOTA performance.

#### 															**Notes**

**Meta learning , *"Learning to learn"*** 

It is a learning method where a model is trained on a variety of tasks in order to get a general or meta-level (higher-level) knowledge about the tasks. Such learning helps the model to uncover the basic, common structural knowledge of the different tasks experienced, which makes the model versatile enough to quickly adapt to the same set of tasks in the future just by experiencing the tasks once (one-shot) or few times (few-shot).  

For instance, a model having a rich, general and holistic knowledge about moving over a surface, can learn to walk, run, hop, crawl and jump over the surface in a shorter amount of time than starting to learn the former mentioned tasks from scratch, by using its basic understanding of locomotion, effectively.

Here, Locomotion (meta-task) --> walk, run, hop,crawl, jump (specific-tasks)

Such a novel learning method can be extended to different domains of machine learning problems like supervised regression & classification and reinforcement learning, where such variants are referred to as different "species" of meta-learning. 

**Model-Agnostic Meta Learning**

***Intuition***    

To perform meta-learning in a way that an optimal meta-learned policy is reached, from which any specific task can be quickly adapted by the model in few gradient steps of learning or with few task -specific experiences.

***Process***  

Meta-learning model is a parameterized function $$f_\theta$$ with parameter $$\theta$$, having a task distribution $$p(\tau)$$ and each task to be learnt is denoted by $$\tau_i$$. In the algorithm, for each task, we first learn the task $$\tau_i$$ using some $$K$$ samples of $$\tau_i$$, and then calculate the adapted parameters with gradient descent. This concludes the training on the *adaptation process*. Using the adapted parameters and an another batch of $$\tau_i$$ , we perform the meta-optimization, which performs gradient step using the expected loss on performance of $$f_\theta$$ over all of the tasks present in the $$p(\tau)$$.

​												$$θ'_i = θ − \alpha∇_{θ}\mathcal{L}_{\tau_i} (f_\theta)$$				 --*adaptation step*

​										$$θ ← θ − β∇_θ\sum_{\tau_{i}∼p(\tau)} \mathcal{L}_{\tau_i} (f_{θ'_i} )$$ 	   --*meta optimization*

The only difference in supervised and reinforcement domains using this algorithm is in the loss function and dataset from with samples of tasks are taken. In supervised learning, the dataset is in the form of $$(x_i,y_i)$$ and loss function is either MSE (for regression problems) or Cross-entropy (for classification problems). 

*MSE :*  $$\mathcal{L}_{\tau_i} (f_\phi) = \sum_{x^{(j)},y^{(j)}∼\tau_i} ||f_\phi(x(j)) − y(j)||_2 ^2 $$ 

*Cross-Entropy Loss :*  $$\mathcal{L}_{\tau_i} (f_\phi) = \sum_{x^{(j)},y^{(j)}∼\tau_i} y^{(j)} \log f_\phi(x^{(j)}) + (1 − y_{(j)} ) \log(1 − f_\phi(x^{(j)} ))$$ 

Whereas, in reinforcement learning, the dataset is the collection of trajectories using $$f_\theta$$ for task $$\tau_i$$ and the loss function, given below, is the negative of expectation of the reward function using the adapted parameters $$\theta_i$$ over all the tasks.

​									$$\mathcal{L}_{\tau_i} (f_\phi) = − \mathbb{E}_{x_t,a_t∼f_\phi,q_{\tau_i}} [ \sum^H_{t=1} R_i(x_t, a_t) ]$$ 

**Experiments**

In an experiment to predict the sinusoidal nature, the model trained using the MAML method was able to infer the ampllitude and phase in the other half of the range, where no datapoints where given in training process, showing that the model has learnt the periodic nature of the sine wave.

in classification domain, the MAML model performed similar to some of the SOTA methods used in few-shot classification, and outperformed memory-augmented networs and the meta-learner LSTM . 

*Interesting fact  :*  Use of a first-order appriximation in the gradientof the meta-objective (First-order MAML or FOMAML), exhibited similar performance   like that of second order gradient present in MAML. This shows that most of the improvement comes from the inner loop of adaptation,more specifically, the adapted parameters, instead of updates in the meta-optimizing outer loop. This approximation speeds up the modle computation by 33%.

In reinforcement learning domain, the inner gradient updates were computed using the vanilla policy gradient (REINFORCE) and for the meta-optimizer trust-region policy optmization (TRPO). 

*Interesting fact :* The authors noticed that halving the learning rate after the first gradient step exhibited superior performance. So, step-size during adaptation was set to $$\alpha = 0.1$$ in the 1^st^ step, then $$\alpha = 0.05$$ for all future steps.

