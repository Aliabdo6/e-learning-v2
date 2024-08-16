### Policy-Based Methods in Reinforcement Learning

---

#### Overview

In this lesson, we will focus on Policy-Based Methods in Reinforcement Learning (RL). Unlike Value-Based Methods, which learn the value of actions or states, Policy-Based Methods directly learn the policy that maps states to actions. We will explore the REINFORCE algorithm and understand how it optimizes policies.

---

#### 1. Introduction to Policy-Based Methods

Policy-Based Methods focus on learning a policy function \( \pi(a|s) \) that represents the probability of taking action \( a \) in state \( s \). The policy can be deterministic or stochastic.

**Key Concepts:**

- **Policy Function \( \pi(a|s) \):** A function that gives the probability of taking action \( a \) given state \( s \).
- **Objective:** Maximize the expected cumulative reward by optimizing the policy function.
- **Gradient Ascent:** Policy-Based Methods use gradient ascent to optimize the policy directly.

**Advantages:**

- Can handle high-dimensional action spaces.
- Can represent stochastic policies.

**Disadvantages:**

- Typically requires more samples to converge.
- May be less stable compared to Value-Based Methods.

---

#### 2. REINFORCE Algorithm

**REINFORCE** is a Monte Carlo-based algorithm for optimizing the policy. It uses the return (cumulative reward) to update the policy parameters.

**Policy Gradient Theorem:**
The gradient of the expected return with respect to the policy parameters \( \theta \) is:
\[ \nabla*{\theta} J(\theta) = \mathbb{E}*{\pi} \left[ \nabla_{\theta} \log \pi(a|s; \theta) \cdot R \right] \]
Where:

- \( J(\theta) \) is the expected return.
- \( \pi(a|s; \theta) \) is the policy function with parameters \( \theta \).
- \( R \) is the return or cumulative reward.

**Algorithm Steps:**

1. Initialize policy parameters \( \theta \).
2. For each episode:
   1. Generate an episode using the current policy.
   2. For each step in the episode, compute the return \( R \).
   3. Update policy parameters \( \theta \) using the policy gradient.

**Python Implementation:**
For this implementation, we'll use a simple policy network and optimize it using the REINFORCE algorithm.

**Step 1: Install Required Libraries**

```bash
pip install gym numpy tensorflow
```

**Step 2: Implement the REINFORCE Algorithm**

```python
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Initialize parameters
gamma = 0.99  # Discount factor
num_episodes = 1000
learning_rate = 0.01

# Create environment
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Define the policy network
model = Sequential([
    Dense(24, activation='relu', input_shape=(n_states,)),
    Dense(24, activation='relu'),
    Dense(n_actions, activation='softmax')
])
optimizer = Adam(learning_rate=learning_rate)

def policy_action(state):
    state = np.expand_dims(state, axis=0)
    probs = model.predict(state)[0]
    return np.random.choice(n_actions, p=probs)

def compute_returns(rewards, gamma):
    returns = np.zeros_like(rewards)
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = rewards[t] + gamma * running_sum
        returns[t] = running_sum
    return returns

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    states, actions, rewards = [], [], []

    done = False
    while not done:
        states.append(state)
        action = policy_action(state)
        next_state, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        state = next_state

    # Compute returns
    returns = compute_returns(rewards, gamma)

    with tf.GradientTape() as tape:
        logits = model(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
        log_probs = tf.math.log(tf.reduce_sum(tf.one_hot(actions, n_actions) * logits, axis=1))
        loss = -tf.reduce_sum(log_probs * tf.convert_to_tensor(returns, dtype=tf.float32))

    # Update policy
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

env.close()
```

---

#### 3. Summary and Next Steps

In this lesson, we covered Policy-Based Methods in Reinforcement Learning, focusing on the REINFORCE algorithm. We discussed its principles, advantages, and implemented it using TensorFlow. In the next lesson, we will explore Actor-Critic Methods, which combine both value-based and policy-based approaches for reinforcement learning.
