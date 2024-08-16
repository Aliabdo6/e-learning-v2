### Actor-Critic Methods in Reinforcement Learning

---

#### Overview

In this lesson, we will explore Actor-Critic Methods in Reinforcement Learning (RL). Actor-Critic methods combine both value-based and policy-based approaches to optimize the policy. We will discuss the principles behind these methods, introduce key algorithms, and implement a basic Actor-Critic algorithm in Python.

---

#### 1. Introduction to Actor-Critic Methods

**Actor-Critic Methods** consist of two components:

- **Actor:** The policy function that suggests actions based on the current state. It is responsible for exploring the action space.
- **Critic:** The value function that evaluates the actions taken by the actor. It provides feedback on the action-value (Q-value) or state-value (V-value).

**Objective:**
The goal is to optimize the policy (actor) using feedback from the value function (critic).

**Advantages:**

- Combines the benefits of value-based and policy-based methods.
- Can stabilize training and improve convergence.

**Disadvantages:**

- May require careful tuning of hyperparameters.
- Complexity increases with the need to train both actor and critic.

---

#### 2. Algorithms: Actor-Critic and A2C

**a. Basic Actor-Critic:**

1. **Initialize** actor and critic networks.
2. **For each episode:**
   1. Generate an episode using the current policy.
   2. **For each step in the episode:**
      1. Compute the advantage function \( A(s, a) \) based on the critic's feedback.
      2. Update the actor's policy using the advantage.
      3. Update the critic's value function using the temporal difference error.

**Advantage Function:**
\[ A(s, a) = Q(s, a) - V(s) \]

**b. Advantage Actor-Critic (A2C):**
A2C is an extension of the basic Actor-Critic algorithm that stabilizes training using generalized advantage estimation (GAE) and synchronizes updates across multiple workers.

---

#### 3. Python Implementation: Basic Actor-Critic

**Step 1: Install Required Libraries**

```bash
pip install gym numpy tensorflow
```

**Step 2: Implement the Actor-Critic Algorithm**

```python
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Initialize parameters
gamma = 0.99  # Discount factor
lambda_ = 0.95  # GAE lambda
num_episodes = 1000
learning_rate = 0.01

# Create environment
env = gym.make('CartPole-v1')
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]

# Define the actor network
actor = Sequential([
    Dense(24, activation='relu', input_shape=(n_states,)),
    Dense(24, activation='relu'),
    Dense(n_actions, activation='softmax')
])
actor_optimizer = Adam(learning_rate=learning_rate)

# Define the critic network
critic = Sequential([
    Dense(24, activation='relu', input_shape=(n_states,)),
    Dense(24, activation='relu'),
    Dense(1)  # Output a single value
])
critic_optimizer = Adam(learning_rate=learning_rate)

def policy_action(state):
    state = np.expand_dims(state, axis=0)
    probs = actor.predict(state)[0]
    return np.random.choice(n_actions, p=probs)

def compute_advantages(rewards, values, gamma, lambda_):
    advantages = np.zeros_like(rewards)
    deltas = rewards + gamma * np.append(values[1:], 0) - values
    running_sum = 0
    for t in reversed(range(len(rewards))):
        running_sum = deltas[t] + gamma * lambda_ * running_sum
        advantages[t] = running_sum
    return advantages

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    states, actions, rewards, values = [], [], [], []

    done = False
    while not done:
        states.append(state)
        action = policy_action(state)
        next_state, reward, done, _ = env.step(action)

        states.append(next_state)
        actions.append(action)
        rewards.append(reward)
        values.append(critic.predict(np.expand_dims(state, axis=0))[0][0])

        state = next_state

    # Compute values for the last state
    values.append(critic.predict(np.expand_dims(state, axis=0))[0][0])

    # Compute advantages
    advantages = compute_advantages(rewards, values, gamma, lambda_)

    # Update actor
    with tf.GradientTape() as tape:
        logits = actor(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
        actions_one_hot = tf.one_hot(actions, n_actions)
        log_probs = tf.math.log(tf.reduce_sum(actions_one_hot * logits, axis=1))
        loss = -tf.reduce_sum(log_probs * tf.convert_to_tensor(advantages, dtype=tf.float32))

    grads = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    # Update critic
    with tf.GradientTape() as tape:
        values_pred = critic(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
        returns = compute_advantages(rewards, values, gamma, lambda_)
        loss = tf.reduce_mean(tf.square(returns - values_pred))

    grads = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

env.close()
```

---

#### 4. Summary and Next Steps

In this lesson, we covered Actor-Critic Methods, focusing on the basic Actor-Critic algorithm and its implementation. We discussed how the actor and critic work together to optimize the policy. In the next lesson, we will explore more advanced Actor-Critic methods, such as Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG).
