import random
from collections import deque
import numpy as np
from bst import FixedSize_BinarySearchTree

class SumTree:

    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[data_idx]
    


class ReplayBuffer(object):
    def __init__(self, buffer_size, random_seed=123, alpha=0.4,beta=0.4):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        self.episode_start_indices = []  # 新增，用于存储每个 episode 的起始索引
        self.current_episode_start_idx = 0  # 新增，当前 episode 的起始索引
        random.seed(random_seed)

        self.sum_tree = SumTree(buffer_size)
        self.alpha = alpha
        self.epsilon = 1e-5
        self.beta = beta
        self.beta_increment = 1e-3

        # self.tree = FixedSize_BinarySearchTree(capacity=buffer_size)
        # self.epsilon = 1e-5
        # self.alpha = alpha
        # self.beta = beta
        # self.beta_increment_per_sampling = 1e-3
        self.base_priority = self.epsilon**self.alpha

        self.new_flag = True
        self.total_distance = 0


    @staticmethod
    def sigmoid(x, scale=1.0):
        """Apply sigmoid function with an adjustable scale."""
        return 1 / (1 + np.exp(-scale * x))
    
    def add(self, state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y):

        experience = (state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y)
        # self.sum_tree.add(max_priority, experience)
        # self.count += 1
        
        if self.new_flag:
            self.total_distance = state[-4]
            self.new_flag = False
        
        # delta_distance = (state[-4] - next_state[-4]) / self.total_distance
        delta_distance = (state[-4] - next_state[-4]) / state[-4]
        safty = np.min(next_state[:-5]) - 0.5
        # delta_theta = np.abs(state[-3] - next_state[-3]) / np.pi
        # delta_theta = next_state[-3]
        # obstacle_distance = np.min(next_state[:-5])
        # penalty= max(0.0, (0.4 - obstacle_distance))

        # if min(next_state[:-5]) <0.4:
        #     delta_distance += -5

        # sigmoid_progress = self.sigmoid(delta_distance) + self.sigmoid(delta_theta, scale=0.5) - self.sigmoid(penalty, scale=2)
        # sigmoid_progress = self.sigmoid(delta_distance) - self.sigmoid(penalty, scale=2)
        sigmoid_progress = 0.5 * self.sigmoid(delta_distance, scale=2) + 0.5 * self.sigmoid(safty)

        # print(delta_distance, sigmoid_progress)

        if self.count < self.buffer_size:
            # self.buffer.append(experience)
            # self.sum_tree.add(max_priority, experience)
            self.sum_tree.add(sigmoid_progress, experience)
            self.count += 1
        else:
            # self.buffer.popleft()
            # self.buffer.append(experience)
            self.sum_tree.add(sigmoid_progress, experience)
            # self.sum_tree.add(max_priority, experience)

        if done:  # 如果当前 experience 标志 episode 结束
            self.episode_start_indices.append(self.current_episode_start_idx)
            self.current_episode_start_idx = self.count  # 更新为下一个 episode 的起始位置
            self.new_flag = True

    def add(self, state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y, max_priority):

        experience = (state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y)
        # self.sum_tree.add(max_priority, experience)
        # self.count += 1
        
        if self.count < self.buffer_size:
            # self.buffer.append(experience)
            # self.sum_tree.add(max_priority, experience)
            self.sum_tree.add(max_priority, experience)
            self.count += 1
        else:
            # self.buffer.popleft()
            # self.buffer.append(experience)
            self.sum_tree.add(max_priority, experience)
            # self.sum_tree.add(max_priority, experience)

        if done:  # 如果当前 experience 标志 episode 结束
            self.episode_start_indices.append(self.current_episode_start_idx)
            self.current_episode_start_idx = self.count  # 更新为下一个 episode 的起始位置

    # def add(self, experiences, success):

    #     for i in range(len(experiences)):
    #         experience = experiences[i]
        
    #     experience = (state, action, reward, terminate, done, next_state, odom_x, odom_y, angle, goal_x, goal_y)
    #     # self.sum_tree.add(max_priority, experience)
    #     # self.count += 1
        
    #     if self.count < self.buffer_size:
    #         # self.buffer.append(experience)
    #         # self.sum_tree.add(max_priority, experience)
    #         self.sum_tree.add(max_priority, experience)
    #         self.count += 1
    #     else:
    #         # self.buffer.popleft()
    #         # self.buffer.append(experience)
    #         self.sum_tree.add(max_priority, experience)
    #         # self.sum_tree.add(max_priority, experience)

    #     if done:  # 如果当前 experience 标志 episode 结束
    #         self.episode_start_indices.append(self.current_episode_start_idx)
    #         self.current_episode_start_idx = self.count  # 更新为下一个 episode 的起始位置
    
    def _get_max_priority(self):
        # try:
        #     max_priority = self.tree.max_value()
        # except:
        #     max_priority = self.base_priority
        max_priority = np.max(self.sum_tree.tree[-self.sum_tree.capacity:])  # Max value from leaf nodes
        if max_priority == 0:
            max_priority = self.base_priority

        return max_priority
    
  
    
    def update_priorities(self,idxs,td_errors):
        new_priorities = np.abs(td_errors)**self.alpha

        #print ("update: {:.2f},{:.2f},{:.2f}".format(self.tree.value_sum,np.max(self.tree.values),np.max(new_priorities)))
        for idx,new_priority in zip(idxs,new_priorities):
            self.sum_tree.update(idx, new_priority)

    def size(self):
        return self.count
    

    def sample_batch(self, batch_size, progress):
        
        # Decay factor for progressive uniform sampling
        alpha = max(0, 1 - progress)  # Decay alpha for priority sampling

        batch = []
        indexes = []
        priorities = []
        priority_segment = self.sum_tree.total() / batch_size

        self.beta = np.min([1., self.beta + self.beta_increment])

        if self.count < batch_size:
            batch_size = self.count
            # print("Batch size is smaller than buffer size, reducing batch size to", batch_size)

        for i in range(batch_size):
            a = i * priority_segment
            b = (i + 1) * priority_segment
            data, index, priority = 0, 0, 0
            while isinstance(data, int):
                # UGLY hack until I find reason why sometimes the tree
                # returns 0 instead of tuple
                value = random.uniform(a, b)
                (index, priority, data) = self.sum_tree.get(value)
            batch.append(data)
            indexes.append(index)
            priorities.append(priority)
            
        # for i in range(int(np.ceil(batch_size*alpha))):
        # #     if np.random.rand() < alpha:
        #     # Priority-based sampling
        #     a = i * priority_segment
        #     b = (i + 1) * priority_segment
        #     value = random.uniform(a, b)
        #     index, priority, data = self.sum_tree.get(value)
        #     # Append the sampled data and info
        #     batch.append(data)
        #     indexes.append(index)
        #     priorities.append(priority / self.sum_tree.total())

        # # Uniform sampling (random index between 0 and count-1)
        # for _ in range(batch_size - len(batch)):
        #     index = np.random.randint(0, self.sum_tree.n_entries - 1)
        #     priority = 1 / self.sum_tree.n_entries  # For uniform sampling, treat priority as equal (e.g., 1)
        #     data = self.sum_tree.data[index]  # Access the stored data directly by index
        #     # Append the sampled data and info
        #     batch.append(data)
        #     indexes.append(index)
        #     priorities.append(priority)


        # # Calculate importance sampling weights
        # # sampling_probabilities = priorities / self.sum_tree.total()
        # # Blend uniform sampling into the probability distribution
        # # sampling_probabilities = alpha * sampling_probabilities + (1 - alpha) * (1 / self.sum_tree.n_entries)
        # priorities[priorities == 0] = 1e-10  # Avoid division by zero
        # importance_sampling_weights = np.power(self.sum_tree.n_entries * priorities, -self.beta)
        # importance_sampling_weights /= importance_sampling_weights.max()  # Normalize

        # print("Priorities:", priorities)
        probabilities = priorities / self.sum_tree.total()
        # assure there is no zeros here
        probabilities[probabilities == 0] = 1e-10
        importance_sampling_weights = np.power(self.sum_tree.n_entries *
                                               probabilities, -self.beta)
        importance_sampling_weights /= importance_sampling_weights.max()

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
        t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
        d_batch = np.array([_[4] for _ in batch]).reshape(-1, 1)
        s2_batch = np.array([_[5] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch, importance_sampling_weights, indexes


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