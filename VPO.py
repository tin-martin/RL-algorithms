import numpy as np
import torch
import torch.nn as nn
import torch.optim
import gymnasium as gym
import itertools

env = gym.make("CartPole-v1")

policy = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n),
        )

optim = torch.optim.Adam(policy.parameters(), lr=1e-3)

observation, _ = env.reset()
T = 1000
gamma = 0.99
# for k = 0,1,2 ... do
total_reward = 0

i = 0
for k in itertools.count():
    traj_s = []
    traj_s_prime = []
    traj_log = []
    traj_reward = []

    # collect set of trajectories by running policy
    for t in range(T):
        i += 1
        obs_t = torch.tensor(observation, dtype=torch.float32)

        logits = policy(obs_t)
        dist = torch.distributions.Categorical(logits=logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        traj_log.append(log_prob)

        new_observation, reward, terminated, truncated, info = env.step(action.item())
        total_reward += reward

        traj_s.append(observation)
        traj_s_prime.append(new_observation)

        traj_reward.append(reward)

        if terminated or truncated:
            observation, _ = env.reset()
            print(k, total_reward)
            total_reward = 0
            break
        else:
            observation = new_observation

    #compute rewards-to-go
    G = 0
    rewards_to_go = []
    for r in reversed(traj_reward):
        G = r + gamma * G
        rewards_to_go.insert(0, G)

    rewards_to_go = torch.tensor(rewards_to_go)
    rewards_to_go = (rewards_to_go - rewards_to_go.mean()) / (rewards_to_go.std() + 1e-8)

    # estimate policy gradient
    traj_log = torch.stack(traj_log, dim=0)
    loss = -torch.mul(traj_log, rewards_to_go).mean()

    optim.zero_grad()
    loss.backward()
    optim.step()

    if i >= 100_000: break

env = gym.make("CartPole-v1", render_mode = "human")
observation, _ = env.reset()
episode_over = False

while not episode_over:
    obs_t = torch.tensor(observation, dtype=torch.float32)

    action = torch.argmax(policy(obs_t))

    if isinstance(action, torch.Tensor):
        action = action.item()

    observation, reward, terminated, truncated, info = env.step(action)

    episode_over = terminated or truncated
    if truncated:
        print("sucess! ")

    env.render()

env.close()


