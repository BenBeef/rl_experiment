import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import time


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


# 定义值函数网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# PPO 算法
class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2, epochs=10, batch_size=64):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.optimizer_policy = optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size

    def act(self, state):
        state = np.array(state)
        state = torch.FloatTensor(state)
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, states, actions, log_probs, returns, advantages):
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(log_probs)

        for _ in range(self.epochs):
            # 更新策略网络
            probs = self.policy(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 更新值函数网络
            # 本质就是做epochs次策略评估(评估状态价值函数), 我们可以假设其收敛了, 那么结果就是当前(执行update之前)策略下状态价值
            values = self.value(states).squeeze()
            value_loss = F.mse_loss(values, returns)

            # 反向传播
            self.optimizer_policy.zero_grad()
            self.optimizer_value.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            self.optimizer_policy.step()
            self.optimizer_value.step()


# 训练 PPO
def train_ppo(env, ppo, episodes=200, max_steps=500):
    for episode in range(episodes):
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        state, _ = env.reset()

        for _ in range(max_steps):
            action, log_prob = ppo.act(state)
            next_state, reward, done, truncated, info = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done or truncated:
                break

        returns = ppo.compute_returns(rewards, dones)
        values = ppo.value(torch.FloatTensor(states)).squeeze().detach()
        advantages = returns - values

        ppo.update(states, actions, log_probs, returns, advantages)

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {sum(rewards)}")


def show_animation(env, ppo, max_steps=1000):
    state, _ = env.reset()
    step = 0
    for _ in range(max_steps):
        env.render()  # 渲染当前状态
        action, _ = ppo.act(state)
        state, _, done, truncated, _ = env.step(action)
        time.sleep(0.02)  # 控制动画速度
        step += 1
        if done or truncated:
            break
    env.close()
    print(f"step:{step}")


# 主程序
if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo = PPO(state_dim, action_dim)
    train_ppo(env, ppo)

    env.close()
    # 展示训练后的动画
    print("Showing trained agent...")
    env = gym.make('CartPole-v1', render_mode="human")  # 重新创建环境，启用渲染
    show_animation(env, ppo)
