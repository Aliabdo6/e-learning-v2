### Multi-Agent Reinforcement Learning (MARL)

---

#### Overview

In this lesson, we will explore Multi-Agent Reinforcement Learning (MARL). Unlike single-agent RL, MARL involves environments where multiple agents interact and potentially compete or cooperate. We will discuss key concepts, challenges, and algorithms used in MARL, and implement a basic MARL setup.

---

#### 1. Introduction to Multi-Agent Reinforcement Learning

**Multi-Agent Reinforcement Learning (MARL)** involves multiple agents interacting in a shared environment. These agents may have their own goals and strategies, which can be competitive, cooperative, or a mix of both.

**Key Concepts:**

- **Centralized vs. Decentralized Training:** Centralized training involves agents learning with access to global information, while decentralized training involves agents learning based only on their local observations.
- **Cooperative vs. Competitive Environments:** Cooperative environments require agents to work together, while competitive environments involve agents competing against each other.
- **Joint Action Spaces:** In MARL, agents often need to consider the joint actions of all agents, making the action space significantly larger.

**Challenges:**

- **Non-Stationarity:** Each agent's policy changes over time, making the environment non-stationary from the perspective of other agents.
- **Scalability:** As the number of agents increases, the complexity of the problem grows exponentially.
- **Credit Assignment:** Determining which agent is responsible for a particular outcome can be challenging.

---

#### 2. Key Algorithms in MARL

**a. Independent Q-Learning (IQL):**

- **Concept:** Each agent independently learns its Q-values assuming other agents' policies are fixed.
- **Challenge:** The non-stationarity of other agents' policies can make learning difficult.

**b. Centralized Training with Decentralized Execution (CTDE):**

- **Concept:** During training, agents have access to global information (centralized training), but during execution, they act based on local information (decentralized execution).
- **Example Algorithms:** Multi-Agent Deep Deterministic Policy Gradient (MADDPG), COMA.

**c. MADDPG (Multi-Agent Deep Deterministic Policy Gradient):**

- **Concept:** An extension of DDPG for multiple agents. Each agent has its own actor and critic network, and the critic is trained with the actions of all agents.

**Algorithm Steps:**

1. **Initialize** actor and critic networks for each agent, along with target networks and experience replay buffers.
2. **For each episode:**
   1. Collect data using the current policies.
   2. Store experiences in the replay buffer.
   3. Sample a batch from the replay buffer.
   4. Update the critic for each agent using the Bellman equation with the actions of all agents.
   5. Update the actor using the policy gradient.
   6. Update target networks.

---

#### 3. Python Implementation: Basic MARL with Independent Q-Learning

We'll create a simple MARL environment using Independent Q-Learning with a two-agent setup in a grid world environment.

**Step 1: Install Required Libraries**

```bash
pip install gym numpy
```

**Step 2: Define the Grid World Environment**

Here's a simplified 2x2 grid world environment with two agents. The agents will learn to move to the goal.

```python
import numpy as np
import gym
from gym import spaces

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.action_space = spaces.Discrete(4)  # Actions: 0: Left, 1: Down, 2: Right, 3: Up
        self.observation_space = spaces.MultiDiscrete([4, 4])  # 4x4 grid
        self.state = [0, 0]  # Initial state

    def reset(self):
        self.state = [0, 0]
        return np.array(self.state)

    def step(self, action):
        if action == 0:  # Left
            self.state[0] = max(0, self.state[0] - 1)
        elif action == 1:  # Down
            self.state[1] = max(0, self.state[1] - 1)
        elif action == 2:  # Right
            self.state[0] = min(3, self.state[0] + 1)
        elif action == 3:  # Up
            self.state[1] = min(3, self.state[1] + 1)

        reward = 1 if self.state == [3, 3] else 0
        done = reward == 1
        return np.array(self.state), reward, done, {}

    def render(self, mode='human'):
        grid = np.zeros((4, 4))
        grid[tuple(self.state)] = 1
        print(grid)

env = GridWorldEnv()
```

**Step 3: Implement Independent Q-Learning for Two Agents**

```python
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_actions, n_states, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.q_table = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(env.action_space.n))
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

# Initialize agents
n_states = env.observation_space.nvec[0] * env.observation_space.nvec[1]
agents = [QLearningAgent(env.action_space.n, n_states) for _ in range(2)]

num_episodes = 1000
for episode in range(num_episodes):
    states = [env.reset() for _ in range(2)]
    done = [False, False]

    while not any(done):
        actions = [agent.choose_action(state[0] * 4 + state[1]) for agent, state in zip(agents, states)]
        next_states, rewards, dones, _ = zip(*[env.step(action) for action in actions])

        for i, agent in enumerate(agents):
            agent.update(states[i][0] * 4 + states[i][1], actions[i], rewards[i], next_states[i][0] * 4 + next_states[i][1])

        states = next_states
        done = dones

env.close()
```

---

#### 4. Summary and Next Steps

In this lesson, we covered Multi-Agent Reinforcement Learning (MARL), including key concepts, challenges, and algorithms such as Independent Q-Learning and MADDPG. We implemented a basic MARL setup using Independent Q-Learning in a grid world environment. In the next lesson, we will explore Hierarchical Reinforcement Learning, which deals with decomposing complex tasks into simpler sub-tasks.
