import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import itertools
from collections import deque

GAMMA = 0.99
BATCH_SIZE = 32
BUFFER_SIZE = 50000
MIN_REPLAY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 10000
TARGET_UPDATE_FREQ = 1000

env = gym.make("MountainCar-v0")


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)



current_network = Network()
target_network = Network()
target_network.load_state_dict(current_network.state_dict())
optimizer = optim.Adam(current_network.parameters(), lr=1e-3)

observation = env.reset()[0]
replay_buffer_s =  np.zeros((BUFFER_SIZE, env.observation_space.shape[0])) #s
replay_buffer_s_prime =  np.zeros((BUFFER_SIZE, env.observation_space.shape[0])) #s'
replay_buffer_a =  np.zeros((BUFFER_SIZE, 1)) #a
replay_buffer_r =  np.zeros((BUFFER_SIZE, 1)) #r
replay_buffer_done =  np.zeros((BUFFER_SIZE, 1))
total_reward = 0
for i in itertools.count():
    epsilon = np.interp(i, [0,EPSILON_DECAY], [EPSILON_START,EPSILON_END])
    if np.random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = torch.argmax(current_network.forward(torch.Tensor(observation))).item()

    new_observation, reward, terminated, truncated, info = env.step(action)


    total_reward += reward
    done = terminated or truncated

    replay_buffer_s[i % BUFFER_SIZE, :] = observation
    replay_buffer_s_prime[i % BUFFER_SIZE, :] = new_observation
    replay_buffer_a[i % BUFFER_SIZE, :] = action
    replay_buffer_r[i % BUFFER_SIZE, :] = reward
    replay_buffer_done[i % BUFFER_SIZE, :] = done

    if done == 1:
        print(total_reward, i)
        total_reward = 0
        observation = env.reset()[0]
    else:
        observation = new_observation

    if i > MIN_REPLAY_SIZE:
        random_indices = np.random.choice(np.min((i,BUFFER_SIZE)), BATCH_SIZE, replace=False)
        replay_buffer_s_t = torch.Tensor(replay_buffer_s[random_indices, :])
        replay_buffer_done_t = torch.Tensor(replay_buffer_done[random_indices, :])
        replay_buffer_r_t = torch.Tensor(replay_buffer_r[random_indices, :])
        replay_buffer_s_prime_t = torch.Tensor(replay_buffer_s_prime[random_indices, :])
        replay_buffer_a_t = torch.Tensor(replay_buffer_a[random_indices, :])

        q_values = current_network.forward(replay_buffer_s_t)
        q_values = q_values.gather(1, replay_buffer_a_t.long()).squeeze(1)

        target_q_values = target_network.forward(replay_buffer_s_prime_t).max(dim=1)[0]


        target = (replay_buffer_r_t + (1 - replay_buffer_done_t) * GAMMA * target_q_values).detach()

        loss = nn.MSELoss()(q_values, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if i % TARGET_UPDATE_FREQ == 0 and i > 0:
        target_network.load_state_dict(current_network.state_dict())

    if i == 250_000:
        break

env = gym.make("MountainCar-v0", render_mode="human")
observation, _ = env.reset()  # reset() returns (obs, info) in Gym ≥0.26
episode_over = False

while not episode_over:
    # convert observation to tensor, add batch dimension
    obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)

    # pick action from your network
    action = torch.argmax(current_network.forward(obs_tensor)).item()

    # if the network outputs tensors or logits, make sure it's an int
    if isinstance(action, torch.Tensor):
        action = action.item()

    # step the environment
    observation, reward, terminated, truncated, info = env.step(action)

    # check if episode is done
    episode_over = terminated or truncated

    env.render()

env.close()









