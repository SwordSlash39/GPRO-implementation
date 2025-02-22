import gymnasium as gym
import torch
import torch.nn as nn
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu_device = torch.device("cpu")

# GPRO
class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.rnn = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, 64),
            nn.Mish(),
            nn.Linear(64, output_size),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.rnn(x)
        return x

# Net
policy = Policy(8, 4)
policy_optim = torch.optim.Adam(policy.parameters(), lr=1e-4)

# Initialise the environment
env = gym.make("LunarLander-v3")

# Training variables
BATCH_SIZE = 64
EPISODES = 1000
EPSILON = 0.2
seed = random.randint(1, 10000)

# Data
"""
Data is BATCH_SIZE x GAME_LEN x (state, old_policy, action, advantage (shld be empty))
"""
data = []
train_data = []
train_rewards = []

# Reset the environment to generate the first observation
# Observation len 4
# action 0 or 1
for e in range(EPISODES):
    seed = random.randint(1, 10000)
    rewards = []
    
    # CPU is faster for batch 1
    policy = policy.to(cpu_device)
    
    with torch.no_grad():
        for b in range(BATCH_SIZE): 
            observation, info = env.reset(seed=seed)
            game_over = False
            batch_data = []
            scores = []
            curr_score = 0
            while not game_over:
                state_data = []
                
                # run policy
                policy_out = policy(torch.tensor(observation, dtype=torch.float32).unsqueeze(0)).squeeze(0)
                
                # Get action
                exp_policy_out = torch.exp(policy_out)
                    
                # Sample randomly to get action
                dist = torch.distributions.Categorical(exp_policy_out)
                action = dist.sample().item()
                
                # Append to state_data
                state_data.append(observation)
                state_data.append(policy_out)
                state_data.append(action)
                state_data.append(-1.0)

                # step (transition) through the environment with the action
                # receiving the next observation, reward and if the episode has terminated or truncated
                observation, reward, terminated, truncated, info = env.step(action)
                curr_score += reward

                # If the episode has ended then we can reset to start a new episode
                if terminated or truncated:
                    game_over = True
                    rewards.append(curr_score)
                
                # Append state_data
                batch_data.append(state_data)
            
            data.append(batch_data)
    
    seed = random.randint(1, 10000)
    
    # Check if batch 1
    if len(train_data) == 0:
        train_data = data
        data = []
        train_rewards = rewards
        rewards = []
        continue

    # Update advantage
    train_rewards = np.array(train_rewards)
    rewards_normalised = (train_rewards - train_rewards.mean()) / (train_rewards.std() + 1e-5)
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            train_data[i][j][3] = rewards_normalised[i]
    
    # Transform train data to trainable data
    states, old_policy, actions, advantages = [], [], [], []
    for i in range(len(train_data)):
        for j in range(len(train_data[i])):
            # Append to
            states.append(train_data[i][j][0])
            old_policy.append(train_data[i][j][1])
            actions.append(train_data[i][j][2])
            advantages.append(train_data[i][j][3])
    
    # Shift policy to gpu
    policy = policy.to(device)
    
    # Convert to tensors
    states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    old_policy_tensor = torch.stack(old_policy, dim=0).to(device)
    actions_tensor = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    advantages_tensor = torch.tensor(advantages, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Calculate policy loss
    r_value = torch.exp(torch.gather(policy(states_tensor), 1, actions_tensor) - torch.gather(old_policy_tensor, 1, actions_tensor).detach())
    clamp_advantages = torch.min(r_value * advantages_tensor, torch.clamp(r_value, 1-EPSILON, 1+EPSILON) * advantages_tensor)
    policy_loss = -clamp_advantages.mean()

    # Train
    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step() 
    
    print(f"Episode {e} - Reward: {train_rewards.mean()}") 
    
    # Move data
    train_data = data
    data = []
    train_rewards = rewards
    rewards = []
    
    
env.close()
