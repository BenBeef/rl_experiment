# coding:utf-8

import gym
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# 超参数
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001


# 定义神经网络
class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 经验回收缓存区
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state), np.array(action), np.array(reward, dtype=np.float32), np.array(next_state),
            np.array(done, dtype=np.uint8))

    def __len__(self):
        return len(self.buffer)


# 初始化环境和模型
env = gym.make("CartPole-v1", render_mode=None)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

policy_net = DQN(state_size, action_size)
target_net = DQN(state_size, action_size)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(capacity=MEMORY_SIZE)

epsilon = EPSILON_START


# 动作选择
def select_action(state, greedy=False):
    if random.random() > epsilon or greedy:
        with torch.no_grad():
            return policy_net(torch.FloatTensor(state)).argmax().item()
    else:
        return random.randint(0, action_size - 1)


def train():
    if len(memory) < BATCH_SIZE:
        return

    state, action, reward, next_state, done = memory.sample(BATCH_SIZE)
    state = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    done = torch.FloatTensor(done)

    q_values = policy_net(state).gather(1, action.unsqueeze(1))  # 计算出每个action的Q值
    next_q_values = target_net(next_state).max(1)[0].detach()
    expected_q_values = reward + (1 - done) * GAMMA * next_q_values

    loss = F.mse_loss(q_values, expected_q_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


num_episodes = 500
for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0

    while True:
        action = select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        memory.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        train()

        if done or truncated:  # 如果任务终止或达到最大步数
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode:{episode}, total_reward:{total_reward}, epsilon:{epsilon:.2f}")
# 训练结束后关闭环境
env.close()

# 开启有动画的环境
env = gym.make("CartPole-v1", render_mode="human")
state, _ = env.reset()
max_steps = 1000
step_count = 0
step_counts = []

print(f"action_size:{action_size}")

for _ in range(10):
    while True:
        env.render()  # 渲染当前状态
        action = select_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        step_count += 1

        if done or truncated or step_count >= max_steps:
            break

    print(f"step_count:{step_count}")
    step_counts.append(step_count)
    state, _ = env.reset()
    step_count = 0

print(f"avg step_count:{sum(step_counts) / len(step_counts)}")
step_counts = []

for _ in range(10):
    while True:
        env.render()  # 渲染当前状态
        action = select_action(state, True)
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state
        step_count += 1

        if done or truncated or step_count >= max_steps:
            break

    print(f"step_count:{step_count}")
    step_counts.append(step_count)
    state, _ = env.reset()
    step_count = 0

print(f"avg step_count:{sum(step_counts) / len(step_counts)}")
env.close()