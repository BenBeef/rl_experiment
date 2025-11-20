import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import time

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# 定义神经网络策略
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = torch.tanh(self.mu(x))  # 输出动作均值
        std = torch.exp(self.log_std)  # 输出动作标准差
        return mu, std


# 定义价值函数网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# PPO 算法
class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2, epochs=10,
                 batch_size=64):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.cnt = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.policy(state)
        self.cnt += 1
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.squeeze(0).numpy(), log_prob.item()

    def compute_gae(self, rewards, masks, values, gamma=0.99, lambda_=0.95):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * masks[t] - values[t]
            gae = delta + gamma * lambda_ * masks[t] * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return returns, advantages

    def update(self, states, actions, log_probs_old, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                state_batch = states[idx]
                action_batch = actions[idx]
                log_prob_old_batch = log_probs_old[idx]
                return_batch = returns[idx]
                advantage_batch = advantages[idx]

                # 更新策略
                mu, std = self.policy(state_batch)
                dist = torch.distributions.Normal(mu, std)
                log_prob_new = dist.log_prob(action_batch).sum(-1)
                ratio = (log_prob_new - log_prob_old_batch).exp()
                surrogate1 = ratio * advantage_batch
                surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                # 更新价值函数
                value_pred = self.value(state_batch).squeeze()
                value_loss = F.mse_loss(value_pred, return_batch)

                # 梯度更新
                self.optimizer_policy.zero_grad()
                self.optimizer_value.zero_grad()
                (policy_loss + value_loss).backward()
                self.optimizer_policy.step()
                self.optimizer_value.step()


# 训练 PPO
def train_ppo(env, ppo, episodes=1000, max_steps=1000):
    for episode in range(episodes):
        state, _ = env.reset()  # gymnasium returns (state, info)
        states, actions, log_probs, rewards, masks, values = [], [], [], [], [], []
        episode_reward = 0

        for step in range(max_steps):
            action, log_prob = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)  # gymnasium returns 5 values
            done = terminated or truncated
            value = ppo.value(torch.FloatTensor(state).unsqueeze(0)).item()

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            masks.append(1 - done)
            values.append(value)

            state = next_state
            episode_reward += reward

            if done:
                break

        # 计算 GAE 和回报
        values.append(ppo.value(torch.FloatTensor(state).unsqueeze(0)).item())
        returns, advantages = ppo.compute_gae(rewards, masks, values, ppo.gamma)

        # 更新模型
        ppo.update(states, actions, log_probs, returns, advantages)

        print(f"Episode {episode + 1}, Reward: {episode_reward}")


# 测试函数
def show_ppo(env, ppo, episodes=10, max_steps=1000):
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action, _ = ppo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward

            if done:
                break

        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")
        time.sleep(0.1)  # 稍微暂停一下，方便观察


# 主函数
if __name__ == "__main__":
    # 训练阶段
    print("Starting training phase...")
    env = gym.make('Ant-v4')  # 训练时不渲染
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    ppo = PPO(state_dim, action_dim)
    train_ppo(env, ppo, episodes=1000)
    env.close()

    # 测试阶段
    print("\nStarting testing phase...")
    env = gym.make('Ant-v4', render_mode="human")  # 测试时渲染
    show_ppo(env, ppo, episodes=10)
    env.close()
