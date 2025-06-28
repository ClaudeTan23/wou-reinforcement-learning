import os
import random
import numpy as np
import gym
import tensorflow as tf

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint, Callback
from rl.processors import Processor


SEED = 42
np.random.seed(SEED)
random.seed(SEED)

ENV_NAME = 'Acrobot-v1'
env = gym.make(ENV_NAME)
_ = env.reset(seed=SEED) 

# Add an environment wrapper for compatibility with keras-rl2
class KerasRL2Wrapper(gym.Wrapper):
    def step(self, action):

        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info


# Apply the wrapper to the environment
env = KerasRL2Wrapper(env)

nb_actions = env.action_space.n
input_shape = env.observation_space.shape

print(f"Environment Name: {ENV_NAME}")
print(f"Observation Space Shape: {input_shape}")
print(f"Action Space Size: {nb_actions}")


# Design the Neural Network Architecture (Q-network)
def build_model(input_shape, nb_actions):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(1,) + input_shape))
    model.add(tf.keras.layers.Dense(64, activation='relu')) 
    model.add(tf.keras.layers.Dense(64, activation='relu')) 
    model.add(tf.keras.layers.Dense(nb_actions, activation='linear')) 
    return model

model = build_model(input_shape, nb_actions)
print("\n--- Summary of the built Q-network model ---")
model.summary()

# Simple Processor class
class CustomProcessor(Processor):
    def process_observation(self, observation):

        if isinstance(observation, tuple):
            observation = observation[0]
        return observation.astype('float32')

    def process_state_batch(self, batch):
        return batch.astype('float32')
    

# Build the DQN Agent
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, nb_steps=50000, value_test=.05)

# SequentialMemory: Stores experiences
memory = SequentialMemory(limit=100000, window_length=1)

dqn = DQNAgent(
    model=model, 
    nb_actions=nb_actions, 
    policy=policy, 
    memory=memory, 
    nb_steps_warmup=1000, 
    gamma=.99, 
    target_model_update=10000,
    train_interval=4, 
    processor=CustomProcessor() 
)


# Compile the Agent
dqn.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae'])

print("\n## DQN Agent compiled successfully ##")



# Train the Agent
NB_TRAINING_STEPS = 5000 # Total number of training steps

log_filename = 'dqn_acrobot_log.json'
weights_filename = 'dqn_acrobot_weights.h5'
checkpoint_weights_filename = 'dqn_acrobot_weights_{step}.h5'

# Define callbacks for training:
# ModelIntervalCheckpoint: Saves model weights at specified intervals.
callbacks = [
    ModelIntervalCheckpoint(checkpoint_weights_filename, interval=1000), 
    FileLogger(log_filename, interval=100)
]

print(f"\n### Starting DQN agent training for {NB_TRAINING_STEPS} steps ##")

dqn.fit(env, nb_steps=NB_TRAINING_STEPS, visualize=False, verbose=2, callbacks=callbacks)
print("\n### Training finished ###")


# Evaluate the Agent
NB_EVALUATION_EPISODES = 10 
print(f"\n### Starting evaluation of the trained agent for {NB_EVALUATION_EPISODES} episodes ###")

history = dqn.test(env, nb_episodes=NB_EVALUATION_EPISODES, visualize=False, verbose=1) 


episode_rewards = history.history['episode_reward']
average_reward = np.mean(episode_rewards)
print(f"Evaluation episode rewards: {episode_rewards}")
print(f"Average reward: {average_reward:.2f}")
print("### Evaluation finished ###")


# Save Model Weights

dqn.save_weights(weights_filename, overwrite=True)
print(f"\nModel weights saved to: {weights_filename}")


# Load and Test Saved Model

loaded_model = build_model(input_shape, nb_actions)
loaded_dqn = DQNAgent(
    model=loaded_model,
    nb_actions=nb_actions,
    policy=policy,
    memory=memory,
    nb_steps_warmup=1000,
    gamma=.99,
    target_model_update=10000,
    train_interval=4,
    processor=CustomProcessor() # <--- Add the processor here
)

# Compile the loaded agent
loaded_dqn.compile(tf.keras.optimizers.legacy.Adam(learning_rate=1e-3), metrics=['mae'])

# Load the weights into the new agent's model
loaded_dqn.load_weights(weights_filename)
print(f"Model weights loaded from {weights_filename}")

# Test the loaded agent
NB_LOADED_TEST_EPISODES = 5
print(f"\nStarting evaluation of the loaded agent for {NB_LOADED_TEST_EPISODES} episodes")

loaded_history = loaded_dqn.test(env, nb_episodes=NB_LOADED_TEST_EPISODES, visualize=False, verbose=1)

loaded_episode_rewards = loaded_history.history['episode_reward']
loaded_average_reward = np.mean(loaded_episode_rewards)
print(f"Loaded agent evaluation episode rewards: {loaded_episode_rewards}")
print(f"Loaded agent average reward: {loaded_average_reward:.2f}")

# Close the environment
env.close()


