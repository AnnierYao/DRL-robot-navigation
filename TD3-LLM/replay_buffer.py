"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque
import numpy as np

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.episode_start_indices = []  # 新增，用于存储每个 episode 的起始索引
        self.current_episode_start_idx = 0  # 新增，当前 episode 的起始索引
        random.seed(random_seed)

    def add(self, s, a, r, terminate, done, next_state):
        experience = (s, a, r, terminate, done, next_state)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

        if done:  # 如果当前 experience 标志 episode 结束
            self.episode_start_indices.append(self.current_episode_start_idx)
            self.current_episode_start_idx = self.count  # 更新为下一个 episode 的起始位置

    def size(self):
        return self.count

   
    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        d_batch = np.array([_[4] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[5] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch


    def get_last_episode(self, i):
        if not self.episode_start_indices:
            raise ValueError("No episode data in buffer")

        start_idx = self.episode_start_indices[-1]
        
        # 检查索引是否越界
        if start_idx + i >= self.count:
            raise IndexError("Index out of bounds for the last episode")

        return self.buffer[start_idx + i]

    def update_last_episode(self, i, s, a, r, terminate, done, next_state):
        if not self.episode_start_indices:
            raise ValueError("No episode data in buffer")

        start_idx = self.episode_start_indices[-1]
        
        # 检查索引是否越界
        if start_idx + i >= self.count:
            raise IndexError("Index out of bounds for the last episode")

        self.buffer[start_idx + i] = (s, a, r, terminate, done, next_state)

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.episode_start_indices.clear()  # 新增，清空 episode 索引
        self.current_episode_start_idx = 0  # 新增，重置当前 episode 索引
