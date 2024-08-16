### Advanced Actor-Critic Methods

---

#### Overview

In this lesson, we will delve into advanced Actor-Critic methods, focusing on Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG). These algorithms address some of the limitations of basic Actor-Critic methods and are widely used in practice for more complex environments.

---

#### 1. Proximal Policy Optimization (PPO)

**Proximal Policy Optimization (PPO)** is an advanced Actor-Critic algorithm designed to improve training stability and efficiency. It aims to optimize the policy while keeping changes within a "trust region" to prevent large, destabilizing updates.

**Key Concepts:**

- **Clipped Objective Function:** PPO uses a clipped surrogate objective function to limit policy updates.
- **Advantage Function:** PPO typically uses the Generalized Advantage Estimation (GAE) for calculating advantages.
- **Surrogate Objective:** PPO optimizes a surrogate objective function to approximate the policy gradient.

**Objective Function:**
The PPO objective function is:
\[ L^{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min \left( \frac{\pi_\theta(a*t | s_t)}{\pi*{\text{old}}(a*t | s_t)} \hat{A}\_t, \text{clip} \left( \frac{\pi*\theta(a*t | s_t)}{\pi*{\text{old}}(a_t | s_t)}, 1 - \epsilon, 1 + \epsilon \right) \hat{A}\_t \right) \right] \]
Where:

- \( \pi\_\theta \) is the current policy.
- \( \pi\_{\text{old}} \) is the previous policy.
- \( \hat{A}\_t \) is the advantage estimate.
- \( \epsilon \) is the clipping parameter.

**Algorithm Steps:**

1. **Initialize** policy and value networks.
2. **For each iteration:**
   1. Collect data using the current policy.
   2. Compute advantages and returns.
   3. Update the policy using the clipped surrogate objective.
   4. Update the value function.

**Python Implementation:**
Here is a simplified implementation of PPO using TensorFlow.

**Step 1: Install Required Libraries**

```bash
pip install gym numpy tensorflow
```

**Step 2: Implement PPO**

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
epsilon = 0.2  # Clipping parameter
learning_rate = 0.001
num_episodes = 1000
batch_size = 64

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

    # Compute advantages and returns
    advantages = compute_advantages(rewards, values, gamma, lambda_)
    returns = compute_returns(rewards, gamma)

    # Update policy
    with tf.GradientTape() as tape:
        logits = actor(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
        actions_one_hot = tf.one_hot(actions, n_actions)
        old_probs = tf.reduce_sum(actions_one_hot * logits, axis=1)
        ratio = old_probs / tf.reduce_sum(actions_one_hot * logits, axis=1)
        clipped_ratio = tf.clip_by_value(ratio, 1 - epsilon, 1 + epsilon)
        loss = -tf.reduce_sum(tf.minimum(ratio * tf.convert_to_tensor(advantages, dtype=tf.float32),
                                         clipped_ratio * tf.convert_to_tensor(advantages, dtype=tf.float32)))

    grads = tape.gradient(loss, actor.trainable_variables)
    actor_optimizer.apply_gradients(zip(grads, actor.trainable_variables))

    # Update critic
    with tf.GradientTape() as tape:
        values_pred = critic(tf.convert_to_tensor(np.array(states), dtype=tf.float32))
        loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(returns, dtype=tf.float32) - values_pred))

    grads = tape.gradient(loss, critic.trainable_variables)
    critic_optimizer.apply_gradients(zip(grads, critic.trainable_variables))

env.close()
```

---

#### 2. Deep Deterministic Policy Gradient (DDPG)

**Deep Deterministic Policy Gradient (DDPG)** is an advanced Actor-Critic algorithm designed for continuous action spaces. It uses both an actor and critic network, with target networks and experience replay to improve stability and efficiency.

**Key Concepts:**

- **Deterministic Policy:** Unlike PPO, DDPG uses a deterministic policy function.
- **Experience Replay:** Stores and samples past experiences to break the correlation between consecutive experiences.
- **Target Networks:** Use target networks to stabilize learning by slowly updating the target networks.

**Algorithm Steps:**

1. **Initialize** actor and critic networks, along with target networks and experience replay buffer.
2. **For each episode:**
   1. Collect data using the current policy.
   2. Store experiences in the replay buffer.
   3. Sample a batch from the replay buffer.
   4. Update the critic using the Bellman equation.
   5. Update the actor using the policy gradient.
   6. Update target networks.

---

#### 3. Summary and Next Steps

In this lesson, we explored advanced Actor-Critic methods, specifically Proximal Policy Optimization (PPO) and Deep Deterministic Policy Gradient (DDPG). We discussed their principles and implemented PPO in Python. In the next lesson, we will explore Multi-Agent Reinforcement Learning, which deals with environments where multiple agents interact.
