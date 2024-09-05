"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque
import numpy as np
import math

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35

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

    def add(self, s, a, r, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y):
        experience = (s, a, r, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y)
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

   
    def sample_batch(self, batch_size, use_her=True, her_ratio=0.8):
        batch = []

        # if self.count < batch_size:
        #     batch = random.sample(self.buffer, self.count)
        # else:
        #     batch = random.sample(self.buffer, batch_size)

        if self.count < batch_size:
            batch_size = self.count
        else:
            batch_size = batch_size
        
        for _ in range(batch_size):
            # 随机选择一个 episode
            episode_idx = random.choice(self.episode_start_indices)
            next_episode_idx = self.episode_start_indices[self.episode_start_indices.index(episode_idx) + 1] if self.episode_start_indices.index(episode_idx) + 1 < len(self.episode_start_indices) else self.count
            episode = list(self.buffer)[episode_idx:next_episode_idx]
            # print('len:', len(episode))
            if len(episode) < 2:
                continue
  
            step_state = np.random.randint(0, len(episode)-1)
            # print('step_state:', step_state)
            state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y = episode[step_state]

            if use_her and (np.random.uniform() <= her_ratio):
                # 从这个 episode 中随机选择一个新的虚拟目标
                step_goal = np.random.randint(step_state+1, len(episode))
                # print('step_goal:', step_goal)
                new_goal_x= episode[step_goal][-5]
                new_goal_y = episode[step_goal][-4]
                # 重新计算 reward 和 state
                next_state, reward, target = self.calculate_reward_and_state(state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                done = 1 if target else int(done)
                terminate = 1 if target else int(terminate)

            # 将 HER 的数据加入到采样数据中
            batch.append([state, action, reward, terminate, done, next_state])


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

    def update_last_episode(self, i, s, a, r, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y):
        if not self.episode_start_indices:
            raise ValueError("No episode data in buffer")

        start_idx = self.episode_start_indices[-1]
        
        # 检查索引是否越界
        if start_idx + i >= self.count:
            raise IndexError("Index out of bounds for the last episode")

        self.buffer[start_idx + i] = (s, a, r, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y)
        # print("Updated last episode at index:", start_idx + i)

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.episode_start_indices.clear()  # 新增，清空 episode 索引
        self.current_episode_start_idx = 0  # 新增，重置当前 episode 索引

    def calculate_reward_and_state(self, state, odom_x, odom_y, angle, new_goal_x, new_goal_y):
        # 获取机器人当前位置
        robot_x = odom_x
        robot_y = odom_y
        robot_theta = angle

        # 计算新目标的距离
        distance = np.linalg.norm([robot_x - new_goal_x, robot_y - new_goal_y])

        # 计算机器人朝向与新目标的相对角度
        skew_x = new_goal_x - robot_x
        skew_y = new_goal_y - robot_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta

        theta = beta - robot_theta
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # 计算奖励
        target = False
        collision = min(state[:-4]) < COLLISION_DIST
        min_laser = min(state[:-4])
        action = [state[-2], state[-1]]

        reward = self.get_sparse_reward(target, collision)
        # reward = self.get_reward(target, collision, [0, 0], min_laser)
        # 更新状态
        robot_state = [distance, theta, state[-2], state[-1]]
        new_state = np.append(state[:-4], robot_state)

        return new_state, reward, target

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2
        
    @staticmethod
    def get_sparse_reward(target, collision):
        if target:
            return 10
        elif collision:
            return -5
        else:
            return -0.1
        