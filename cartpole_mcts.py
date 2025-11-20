import gymnasium as gym
import numpy as np
from collections import defaultdict
import math


# 定义 MCTS 节点类
class Node:
    def __init__(self, state, parent=None):
        self.state = state  # 当前状态（CartPole 的观测值）
        self.parent = parent  # 父节点
        self.children = []  # 子节点列表（每个子节点是 (action, node) 的元组）
        self.visits = 0  # 节点访问次数
        self.total_reward = 0.0  # 累计奖励
        self.is_terminal = False  # 是否为终止状态

    def expand(self, env):
        """扩展子节点：尝试所有可能的动作"""
        for action in [0, 1]:  # CartPole 的动作空间为 [左, 右]
            # 复制环境以避免修改原始状态
            copied_env = gym.make("CartPole-v1")
            copied_env.reset()
            copied_env.unwrapped.state = self.state  # 设置当前状态
            # 执行动作
            next_state, reward, done, _, _ = copied_env.step(action)
            child = Node(next_state, parent=self)
            child.is_terminal = done
            self.children.append((action, child))

    def best_child(self, exploration_weight=1.0):
        """选择最佳子节点（UCB 公式）"""
        if not self.children:
            return None
        # 计算每个子节点的 UCB 值，添加极小值避免除零
        ucb_values = [
            (child.total_reward / (child.visits + 1e-5)) +  # 修复：分母加 1e-5
            exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-5))
            for _, child in self.children
        ]
        return self.children[np.argmax(ucb_values)]


# 定义 MCTS 类
class MCTS:
    def __init__(self, num_simulations=100, max_depth=50):
        self.num_simulations = num_simulations  # 每次决策的模拟次数
        self.max_depth = max_depth  # 单次模拟的最大深度

    def search(self, root_state):
        """从初始状态开始搜索"""
        root = Node(root_state)
        for _ in range(self.num_simulations):
            node = root
            depth = 0
            # 1. 选择阶段：找到未完全扩展的节点
            while node.children and depth < self.max_depth:
                action, node = node.best_child()
                depth += 1
            # 2. 扩展阶段：如果未终止且未完全扩展，则扩展
            if not node.is_terminal and not node.children:
                node.expand(gym.make("CartPole-v1"))  # 传递环境实例
            # 3. 模拟阶段：随机策略模拟到终止状态
            total_reward = self.rollout(node.state, depth)
            # 4. 反向传播阶段：更新路径上的节点统计信息
            self.backpropagate(node, total_reward)
        # 选择访问次数最多的动作
        if not root.children:
            return 0  # 默认动作
        best_action = max(root.children, key=lambda x: x[1].visits)[0]
        return best_action

    def rollout(self, state, current_depth):
        """随机策略模拟直到终止或达到最大深度"""
        total_reward = 0.0
        env = gym.make("CartPole-v1")
        env.reset()
        env.unwrapped.state = state  # 设置初始状态
        for _ in range(self.max_depth - current_depth):
            action = env.action_space.sample()  # 随机动作
            _, reward, done, _, _ = env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def backpropagate(self, node, reward):
        """反向传播奖励"""
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node = node.parent


# 主程序：使用 MCTS 控制 CartPole
def main():
    # env = gym.make("CartPole-v1", render_mode="human")
    env = gym.make("CartPole-v1")
    observation, _ = env.reset()
    mcts = MCTS(num_simulations=50, max_depth=500)
    total_reward = 0

    total_step = 0
    while True:
        action = mcts.search(observation)
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        total_step += 1
        if total_step % 10 == 0:
            print(f"total_step:{total_step}")
        if done:
            print(f"Episode finished. Total reward: {total_reward}")
            break


if __name__ == "__main__":
    main()
