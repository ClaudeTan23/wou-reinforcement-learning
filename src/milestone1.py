import sys
import tensorflow as tf
import keras
import gym
import numpy as np
import time

print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")
print(f"Gym version: {gym.__version__}")




env = gym.make("Acrobot-v1", render_mode="rgb_array")

observation_space = env.observation_space

print(f"  Shape: {observation_space.shape}")
print(f"  Low values: {observation_space.low}")
print(f"  High values: {observation_space.high}")

action_space = env.action_space
print(f"\nAction Space: {action_space}")
print(f"  Number of actions: {action_space.n}")
# Reset the environment
initial_state, info = env.reset()
print(f"\nInitial state after reset: {initial_state}")

# Close the environment 
env.close()


# Recreate the environment for observing random actions
env = gym.make("Acrobot-v1", render_mode="human")

num_random_episodes = 3 
max_steps_per_episode = 200 

print(f"Running {num_random_episodes} random episodes:")

for episode in range(num_random_episodes):
    state, info = env.reset() 
    done = False
    truncated = False 
    total_reward = 0
    step_count = 0

    print(f"\n###  Episode {episode + 1}  ###")
    while not done and not truncated and step_count < max_steps_per_episode:
        action = env.action_space.sample() 
        
        # Get the new state, reward, done flag, truncated flag, and additional info
        next_state, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        # Render the environment
        env.render()
        
        # Update the current state
        state = next_state 


    print(f"Episode {episode + 1} finished after {step_count} steps. Total Reward: {total_reward:.2f}")

# Close the environment
env.close()
print("\nFinished running random episodes.")



# Define input and output dimensions based on the Acrobot environment
state_size = 6  
action_size = 3 

# Define the model building function for the Q-network
def build_q_network(input_shape, output_size):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape), 
        tf.keras.layers.Dense(64, activation="relu", name="hidden_layer_1"), 
        tf.keras.layers.Dense(64, activation="relu", name="hidden_layer_2"), 
        tf.keras.layers.Dense(output_size, activation="linear", name="output_layer")
    ])
    return model

# Instantiate the Q-network model
q_network = build_q_network(input_shape=(state_size,), output_size=action_size)

print("Neural Network Architecture Designed Successfully.")


state_size = 6
action_size = 3

# Build the Q-network model using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.Input(shape=(state_size,)), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(64, activation='relu'), 
    tf.keras.layers.Dense(action_size, activation='linear') 
])

print("Keras Neural Network Model Implemented Successfully.")


# Print a summary of the model architecture
model.summary()