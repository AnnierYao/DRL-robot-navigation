"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque
import numpy as np
from bst import FixedSize_BinarySearchTree

class Trajectory:
    def __init__(self, init_state):
        self.states = [init_state]
        self.actions = []
        self.rewards = []
        self.terminates = []
        self.dones = []
        self.next_states = []
        self.odom_xs = []
        self.odom_ys = []
        self.angles = []
        self.goal_xs = []
        self.goal_ys = []
        # self.prioritys = 0
        self.energy_transition = 0
        self.length = 0
    
    def store_step(self, state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y):
        self.actions.append(action)
        self.states.append(state)
        self.rewards.append(reward)
        self.terminates.append(terminate)
        self.dones.append(done)
        self.next_states.append(next_state)
        self.odom_xs.append(odom_x)
        self.odom_ys.append(odom_y)
        self.angles.append(angle)
        self.goal_xs.append(goal_x)
        self.goal_ys.append(goal_y)
        self.length += 1

class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123, alpha=0.4,beta=0.4):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.length = 0
        self.buffer = deque()
        self.episode_start_indices = []  # 新增，用于存储每个 episode 的起始索引
        self.current_episode_start_idx = 0  # 新增，当前 episode 的起始索引
        random.seed(random_seed)

        self.tree = FixedSize_BinarySearchTree(capacity=buffer_size)
        self.epsilon = 1e-5
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = 1e-3
        self.base_priority = self.epsilon**self.alpha

    
    def add_trajectory(self, trajectory):

        linear_velocity = [action[0] for action in trajectory.actions]
        angular_velocity = [action[1] for action in trajectory.actions]

        m, inertia = 1, 1

        kinetic_energy = 0.5 * m * np.power(np.array(linear_velocity),2)
        rotational_energy = 0.5 * inertia * np.power(np.array(angular_velocity),2)

        energy_total = kinetic_energy + rotational_energy

        energy_diff = np.diff(energy_total)
        energy_transition = energy_total.copy()
        energy_transition[1:] = energy_diff
        energy_transition = np.clip(energy_transition,0,999)
        trajectory.energy_transition = np.sum(energy_transition)/trajectory.length
        print(trajectory.energy_transition)

        self.tree.add(trajectory.energy_transition)

        self.count += trajectory.length
        self.length += 1

        if self.count < self.buffer_size:
            self.buffer.append(trajectory)
        else:
            left = self.buffer.popleft()
            self.buffer.append(trajectory)
            self.count -= left.length
    
    # def add(self, s, a, r, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y, max_priority):

    #     m, inertia = 1, 1

    #     linear_velocity = a[0]
    #     angular_velocity = a[1]

    #     kinetic_energy = 0.5 * m * linear_velocity**2
    #     rotational_energy = 0.5 * inertia * angular_velocity**2

    #     energy_total = kinetic_energy + rotational_energy

    #     self.tree.add(max_priority)   

    #     experience = (s, a, r, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y, max_priority)
    #     if self.count < self.buffer_size:
    #         self.buffer.append(experience)
    #         self.count += 1
    #     else:
    #         self.buffer.popleft()
    #         self.buffer.append(experience)

    #     if done:  # 如果当前 experience 标志 episode 结束
    #         self.episode_start_indices.append(self.current_episode_start_idx)
    #         self.current_episode_start_idx = self.count  # 更新为下一个 episode 的起始位置
 
    
    def _get_max_priority(self):
        try:
            max_priority = self.tree.max_value()
        except:
            max_priority = self.base_priority

        return max_priority
    
    def update_priorities(self,idxs,td_errors):
        new_priorities = np.abs(td_errors)**self.alpha

        #print ("update: {:.2f},{:.2f},{:.2f}".format(self.tree.value_sum,np.max(self.tree.values),np.max(new_priorities)))
        for idx,new_priority in zip(idxs,new_priorities):
            self.tree.update(new_priority,idx)

    def size(self):
        return self.count

   
    def sample_batch(self, batch_size):
        batch = dict(states=[], actions=[], next_states=[], rewards=[], terminates=[])

        sampling_probabilities = np.array(self.tree.values)/self.tree.value_sum

        if self.count < batch_size:
            # idxes = np.random.choice(range(self.tree.size),self.count,replace=False,p=sampling_probabilities)
            traj_idxes = np.random.choice(range(self.length),self.count,replace=True,p=sampling_probabilities)
        else:
            traj_idxes = np.random.choice(range(self.length),batch_size,replace=True,p=sampling_probabilities)
    
        sampling_probabilities = sampling_probabilities[traj_idxes]
        is_weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        step_idxes = [np.random.randint(self.buffer[i].length) for i in traj_idxes]

        for traj_idx, step_idx in zip(traj_idxes, step_idxes):
            traj = self.buffer[traj_idx]
            state = traj.states[step_idx]
            next_state = traj.next_states[step_idx]
            action = traj.actions[step_idx]
            reward = traj.rewards[step_idx]
            terminate = traj.dones[step_idx]

            batch['states'].append(state)
            batch['actions'].append(action)
            batch['next_states'].append(next_state)
            batch['rewards'].append(reward)
            batch['terminates'].append(terminate)
        
        # Convert lists to NumPy arrays and return
        batch['states'] = np.array(batch['states'])
        batch['actions'] = np.array(batch['actions'])
        batch['rewards'] = np.array(batch['rewards'])
        batch['next_states'] = np.array(batch['next_states'])
        batch['terminates'] = np.array(batch['terminates'])

        # increment beta
        self.beta = min(1.0, self.beta+self.beta_increment_per_sampling)


        # if self.count < batch_size:
        #     batch = random.sample(self.buffer, self.count)
        # else:
        #     batch = random.sample(self.buffer, batch_size)

        # s_batch = np.array([_[0] for _ in batch])
        # a_batch = np.array([_[1] for _ in batch])
        # r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        # t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        # d_batch = np.array([_[4] for _ in batch]).reshape(-1, 1)
        # s2_batch = np.array([_[5] for _ in batch])


        return batch['states'], batch['actions'], batch['rewards'], batch['terminates'], batch['next_states'], is_weights, traj_idxes


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

    def remove_last_episode_elements(self, num_elements):
        """
        移除最后一个 episode 中的固定数量的元素
        """
        if not self.episode_start_indices:
            raise ValueError("No episode data in buffer")

        # 获取最后一个 episode 的起始索引
        start_idx = self.episode_start_indices[-1]
        
        # 检查需要移除的元素数量是否超出该 episode 的长度
        num_elements_to_remove = min(num_elements, self.count - start_idx)

        # 移除最后一个 episode 中的指定数量的元素
        for _ in range(num_elements_to_remove):
            self.buffer.pop()
            self.count -= 1

        # 如果整个 episode 被移除，则删除其起始索引
        if start_idx >= self.count:
            self.episode_start_indices.pop()

    def clear(self):
        self.buffer.clear()
        self.count = 0
        self.episode_start_indices.clear()  # 新增，清空 episode 索引
        self.current_episode_start_idx = 0  # 新增，重置当前 episode 索引
