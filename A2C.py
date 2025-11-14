

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import gymnasium as gym
import itertools

env = gym.make("CartPole-v1")

policy_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

value_network = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

optim_policy = torch.optim.Adam(policy_network.parameters(), lr=1e-3)
optim_value = torch.optim.Adam(value_network.parameters(), lr=1e-3)

observation, _ = env.reset()
T =10
gamma = 0.99

total_reward = 0
traj_s = []
traj_s_prime = []
traj_log = []
traj_reward = []
traj_done = []

for k in range(100_000):
    obs_t = torch.tensor(observation, dtype=torch.float32)

    logits = policy_network(obs_t)
    dist = torch.distributions.Categorical(logits=logits)

    action = dist.sample()
    log_prob = dist.log_prob(action)

    traj_log.append(log_prob)

    new_observation, reward, terminated, truncated, info = env.step(action.item())
    total_reward += reward

    traj_s.append(observation)
    traj_s_prime.append(new_observation)
    traj_reward.append(reward)
    traj_done.append(terminated)

    if terminated or truncated:
        observation, _ = env.reset()
        print(k, total_reward)
        total_reward = 0
    else:
        observation = new_observation

    if len(traj_reward) >= T or terminated or truncated:
        traj_log = torch.stack(traj_log, dim=0).unsqueeze(1)
        traj_s = torch.tensor(traj_s, dtype=torch.float32)
        traj_s_prime = torch.tensor(traj_s_prime, dtype=torch.float32)
        traj_reward = torch.tensor(traj_reward).unsqueeze(1)
        traj_done = torch.tensor(traj_done).unsqueeze(1)

        advantages = traj_reward + (traj_done).logical_not() * gamma * value_network(traj_s_prime) - value_network(traj_s)
        advantages = advantages.detach()

        loss_policy = -torch.mul(traj_log, advantages).mean()
        optim_policy.zero_grad()
        loss_policy.backward()
        optim_policy.step()

        with torch.no_grad():
            target = traj_reward + (traj_done.logical_not()) * gamma * value_network(traj_s_prime)

        value_pred = value_network(traj_s)
        loss_value = nn.MSELoss()(value_pred, target)
        optim_value.zero_grad()
        loss_value.backward()
        optim_value.step()

        traj_s = []
        traj_s_prime = []
        traj_log = []
        traj_reward = []
        traj_done = []

env = gym.make("CartPole-v1", render_mode = "human")
observation, _ = env.reset()
episode_over = False

while not episode_over:
    obs_t = torch.tensor(observation, dtype=torch.float32)

    action = torch.argmax(policy_network(obs_t))

    if isinstance(action, torch.Tensor):
        action = action.item()

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
    if truncated:
        print("success! ")

    env.render()

env.close()


