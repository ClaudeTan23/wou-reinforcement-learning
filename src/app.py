import numpy as np
import random

# GridWorld environment
class GridWorld:
    def __init__(self, size=4):
        self.size = size
        self.start_state = (0, 0)
        self.goal_state = (size - 1, size - 1)
        self.reset()

    def reset(self):
        self.agent_pos = self.start_state
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and x > 0: x -= 1       # Up
        elif action == 1 and x < self.size - 1: x += 1  # Down
        elif action == 2 and y > 0: y -= 1       # Left
        elif action == 3 and y < self.size - 1: y += 1  # Right
        
        self.agent_pos = (x, y)
        reward = 10 if self.agent_pos == self.goal_state else -1
        done = self.agent_pos == self.goal_state
        return self.agent_pos, reward, done

# Q-learning agent
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 4))  # 4 actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        x, y = state
        return np.argmax(self.q_table[x, y])  # Exploit

    def update_q(self, state, action, reward, next_state):
        x, y = state
        nx, ny = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[nx, ny])
        self.q_table[x, y, action] += self.alpha * (target - predict)

# Training the agent
def train_agent(episodes=500):
    env = GridWorld()
    agent = QLearningAgent(env)
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        while not done and steps < 50:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q(state, action, reward, next_state)
            state = next_state
            steps += 1
    return agent

# Test the agent's learned policy
def test_agent(agent):
    env = agent.env
    state = env.reset()
    done = False
    path = [state]
    while not done:
        action = agent.choose_action(state)
        next_state, _, done = env.step(action)
        path.append(next_state)
        state = next_state
    print("Agent path from start to goal:", path)

# Run simulation
if __name__ == "__main__":
    trained_agent = train_agent()
    test_agent(trained_agent)
