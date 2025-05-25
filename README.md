# 🧠 Reinforcement Learning: GridWorld Q-Learning Simulation

A simple Python-based simulation demonstrating Reinforcement Learning using **Q-Learning** in a 4x4 GridWorld environment. The agent learns to reach the goal efficiently by maximizing rewards through trial-and-error interactions.

---

## 📌 Overview

This project showcases key Reinforcement Learning (RL) concepts:
- An agent navigating an environment
- A reward function guiding behavior
- Q-learning algorithm for policy learning

The goal: **Get from the top-left to the bottom-right cell with as few steps as possible.**

---

## 🗺️ Environment: GridWorld

- **Grid Size**: 4x4
- **Start State**: `(0, 0)`
- **Goal State**: `(3, 3)`
- **Actions**:
  - `0`: Up
  - `1`: Down
  - `2`: Left
  - `3`: Right

---

## 🎯 Reward Function

| Condition           | Reward |
|---------------------|--------|
| Reach Goal          | +10    |
| Any Other Move      | -1     |

This encourages the agent to:
- Reach the goal quickly
- Avoid unnecessary or looping steps

---

## 🚀 How It Works

### Agent: `QLearningAgent`
- **Algorithm**: Q-learning
- **Policy**: ε-greedy (balance of exploration & exploitation)
- **Q-table**: Stores estimated values for each state-action pair

### Training
- Runs multiple episodes to update Q-values
- Learns the best policy to maximize total reward

### Testing
- After training, the agent uses the learned policy to navigate the grid from start to goal.

---

## 🧪 Example Output

```bash
Agent path from start to goal:
[(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]
