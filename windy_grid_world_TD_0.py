# coding:utf-8
import numpy as np
import random

grid_word = np.zeros((7, 10), dtype=np.int32)
m, n = grid_word.shape[0], grid_word.shape[1]

start = (3, 0)
end = (3, 7)
grid_word[start] = 1

winds = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

up = 0
down = 1
left = 2
right = 3
action_space = [up, down, left, right]


def env_step(s, a):
    """
    :param s:
    :param a:
    :return:
    """
    i, j = s[0], s[1]
    w = winds[j]
    if a == up:
        i -= 1
    elif a == down:
        i += 1
    elif a == left:
        j -= 1
    else:
        j += 1
    i = i - w
    i = max(0, i)
    i = min(m - 1, i)
    j = max(0, j)
    j = min(n - 1, j)
    s = (i, j)
    return s, 1, s == end


print(env_step((0, 4), 0))

Q = np.zeros((m, n, len(action_space)), dtype=np.float32)

epsilon = 0.1
alpha = 0.5


def select_action(s):
    if random.random() > epsilon:
        return np.argmin(Q[s])
    else:
        return random.randint(0, len(action_space) - 1)


array = []


def td_0():
    for i in range(1000):
        count = 0
        state = start
        action = select_action(state)
        while True:
            next_state, reward, done = env_step(state, action)
            if done:
                break
            count += 1
            action_next = select_action(next_state)
            qsa = Q[state[0], state[1], action]
            qsa_next = Q[next_state[0], next_state[1], action_next]
            v = qsa + alpha * (reward + qsa_next - qsa)
            Q[state[0], state[1], action] = v
            state, action = next_state, action_next

        array.append(count)
        print(f"Episode:{i + 1} step:", count)

td_0()

import matplotlib.pyplot as plt

# 定义数组


# 生成大量数据点以确保光滑性
x = [i for i in range(len(array))]
y = array

# 绘制光滑图
plt.plot(x, y, label='y = sin(x)', color='b', linestyle='-', linewidth=2)

# 添加标题和标签
plt.title("Smooth Sine Function")  # 标题
plt.xlabel("x")                    # x 轴标签
plt.ylabel("y")                    # y 轴标签

# 显示网格
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图例
plt.legend()

# 显示图形
plt.show()