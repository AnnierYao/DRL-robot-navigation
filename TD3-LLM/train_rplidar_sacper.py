import os
import time

import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.distributions import Normal
from numpy import inf
import datetime
from torch.utils.tensorboard import SummaryWriter


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
# device = torch.device("cpu")
seed = 1  # Random seed number
eval_freq = 1e3  # After how many steps to perform the evaluation
start_steps = 1e4  # Number of steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 8e5  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 256 # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1000000  # Maximum size of the buffer
name = "SAC_rplidar_mixed"  # name of the file to store the policy
save_model = True # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not
reward_type = 'hybird'  # Reward type, sparse or dense
env_name = "RplidarEnv"  # Environment name
# from velodyne_env_maze import GazeboEnv
from rplidar_env import GazeboEnv
# from rplidar_env_maze import GazeboEnv
# from rplidar_env_distance import GazeboEnv
# from rplidar_env_sparse import GazeboEnv
use_LLM_HER = False # Weather to use LLM HER or not
curriculum_steps = 150  # Number of steps to perform the HER algorithm
feedback_form = 'goal'  # Feedback form for the HER algorithm
from gpt_feedback import GoalPredictor
use_HER = False  # Whether to use HER or not
use_TDPER = False  # Whether to use PER HER or not
use_PQPER = True # Whether to use PQPER or not
use_PPER = False  # Whether to use PPER or not
# from replay_buffer import ReplayBuffer
# from replay_buffer_PER import ReplayBuffer
# from replay_buffer_her import ReplayBuffer
# from replay_buffer_PQPER import ReplayBuffer, Trajectory
# from replay_buffer_PPER import ReplayBuffer
from replay_buffer_PQPER import ReplayBuffer
lr = 0.00005  # Learning rate for the networks
alpha = 0.2  # Alpha value for the SAC algorithm 
hidden_dim = 500
file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(name, reward_type, feedback_form if use_LLM_HER else "", env_name,batch_size,lr, hidden_dim,
                                                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                    "HER" if use_HER else "", 
                                                    "LLM_HER" if use_LLM_HER else "", curriculum_steps if use_LLM_HER else "","TDPER" if use_TDPER else "", "PQPER" if use_PQPER else "", "PPER" if use_PPER else "")
result_path = './runs/' + file_name
writer = SummaryWriter(result_path)

api_base = "https://eastusengine.openai.azure.com/"
api_key = '9ae06f2709144320b0b4645c587a3492'
deployment_name = 'gpt-4o-2'
api_version = '2023-03-15-preview'
# api_base = "https://eastusengine.openai.azure.com/"
# api_key = '9ae06f2709144320b0b4645c587a3492'
# deployment_name = 'gpt-4o-mini'
# api_version = '2023-03-15-preview'

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    success = 0
    for _ in range(eval_episodes):
        count = 0
        state, info = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, target, info = env.step(a_in)
            avg_reward += reward
            count += 1
            if target:
                success += 1   
            if info['collision']:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    avg_success = success / eval_episodes   
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f, success rate: %f"
        % (eval_episodes, epoch, avg_reward, avg_col, avg_success)
    )
    print("..............................................")
    writer.add_scalar('test/reward', avg_reward, epoch)
    writer.add_scalar('test/collision', avg_col, epoch) 
    writer.add_scalar('test/success', avg_success, epoch)   
    return avg_reward

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=800, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        # 计算动作熵 H(π(s))
        entropy = normal.entropy().sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean, entropy

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=800):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2



class SAC(object):
    def __init__(self, state_dim, action_dim, alpha=0.2, hidden_dim=800, lr = 0.001):
        # Actor and Critic networks
        # self.max_action = max_action
        self.iter_count = 0
        self.alpha = alpha

        self.critic = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.hard_update(self.critic_target, self.critic)

        self.target_entropy = -torch.prod(torch.Tensor(action_dim).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = Adam([self.log_alpha], lr=lr)

        self.policy = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        # # 使用 StepLR 调度器，每隔 50 次迭代将学习率下降0.8倍
        # self.critic_scheduler = StepLR(self.critic_optim, step_size=50, gamma=0.8)
        # self.policy_scheduler = StepLR(self.policy_optim, step_size=50, gamma=0.8)

        # # 使用 ReduceLROnPlateau 调度器
        # self.critic_scheduler = ReduceLROnPlateau(self.critic_optim, mode='min', factor=0.8, patience=10)
        # self.policy_scheduler = ReduceLROnPlateau(self.policy_optim, mode='min', factor=0.8, patience=10)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        if evaluate is False:
            action, _, _, _ = self.policy.sample(state)
        else:
            _, _, action, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]


    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, progress=0):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_alpha_loss = 0
        av_entropy = 0
        av_alpha = 0
        av_td_error = 0
        av_delta_distance = 0
        for it in range(iterations):
            # Sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
                batch_is_weights,
                idxes,
            ) = replay_buffer.sample_batch(batch_size, progress)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            is_weights = torch.Tensor(batch_is_weights).to(device)

            with torch.no_grad():
                next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_state)
                qf1_next_target, qf2_next_target = self.critic_target(next_state, next_state_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
                next_q_value = reward + (1-done) * discount * (min_qf_next_target)
                av_Q += torch.mean(min_qf_next_target)
                max_Q = max(max_Q, torch.max(min_qf_next_target))
            
            qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
            # 计算 TD 误差
            td_error_qf1 = qf1 - next_q_value
            td_error_qf2 = qf2 - next_q_value

            # 计算加权平方误差
            weighted_squared_td_error1 = is_weights * (td_error_qf1 ** 2)
            weighted_squared_td_error2 = is_weights * (td_error_qf2 ** 2)

            # 计算加权 MSE 损失
            qf1_loss = weighted_squared_td_error1.mean()
            qf2_loss = weighted_squared_td_error2.mean()
            # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss
                
            # qf1, qf2 = self.critic(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
            # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf_loss = qf1_loss + qf2_loss

            # print(type(idxes[0]))
            # print(type(td_error.detach().cpu().numpy().squeeze()))  # 输出类型
            # print(td_error.detach().cpu().numpy().squeeze())        # 输出值

            self.critic_optim.zero_grad()
            qf_loss.backward()
            self.critic_optim.step()
            # self.critic_scheduler.step()
            # 根据验证损失更新学习率
            # self.critic_scheduler.step(qf1_loss)  # 或者使用其他验证损失

            pi, log_pi, _, entropy = self.policy.sample(state)

            qf1_pi, qf2_pi = self.critic(state, pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

            td_error_policy = self.alpha * log_pi - min_qf_pi

            policy_loss = td_error_policy.mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            if use_TDPER:
                td_error = abs(((td_error_qf1 + td_error_qf2) / 6.0 + td_error_policy + 1e-5).squeeze())
                replay_buffer.update_priorities(idxes,td_error.detach().cpu().numpy().squeeze())

            if use_PQPER:
                # alpha = torch.sigmoid(torch.tensor(entropy - entropy_threshold))
                # alpha = torch.sigmoid((entropy - 1).clone().detach()).item()  # 使用 .item() 获取标量
                
                alpha = torch.sigmoid(entropy.mean() - 0.5).item()  # 使用 .item() 获取标量
                # 计算当前策略的熵
                # 计算 delta_distance
                state_distances_to_goal = state[:, -4]  # 取出 state[-4] 距离
                next_state_distances_to_goal = next_state[:, -4]  # 取出 next_state[-4] 距离
                # delta_distances = (state_distances_to_goal - next_state_distances_to_goal) / (state_distances_to_goal + 1e-6)
                delta_distances = state_distances_to_goal - next_state_distances_to_goal

                # # 计算 safty
                # min_laser_distances = torch.min(next_state[:, :-5], dim=1).values  # next_state[:-5] 的最小值
                # safties = min_laser_distances - 0.5  # 安全性计算

                # 计算TD error
                td_error = abs(((td_error_qf1 + td_error_qf2) / 6.0 + td_error_policy + 1e-5).squeeze())

                # Sigmoid 函数应用
                delta_sigmoid = torch.sigmoid(10 * delta_distances) 
                # safty_sigmoid = torch.sigmoid(safties)
                td_error_sigmoid = torch.sigmoid(td_error)

                # 自适应优先级计算
                # priorities = alpha * delta_sigmoid + (1 - alpha) * safty_sigmoid
                priorities = alpha * td_error_sigmoid + (1 - alpha) * delta_sigmoid
                # print('entropy:', entropy.mean().item(), 'alpha:', alpha, 'td_error_sigmoid:', td_error_sigmoid.mean().item(), 'delta_sigmoid:', delta_sigmoid.mean().item())
                av_entropy += entropy.mean().item()
                av_alpha += alpha
                av_td_error += td_error_sigmoid.mean().item()
                av_delta_distance += delta_sigmoid.mean().item()
                replay_buffer.update_priorities(idxes,priorities.detach().cpu().numpy().squeeze())
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()
            # 更新学习率
            # self.policy_scheduler.step()
            # self.policy_scheduler.step(policy_loss)  # 或者使用其他验证损失

            # alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            # self.alpha_optim.zero_grad()
            # alpha_loss.backward()
            # self.alpha_optim.step()
            # self.alpha = self.log_alpha.exp()
            # alpha_tlogs = self.alpha.clone() # For TensorboardX logs

            self.soft_update(self.critic_target, self.critic, tau)


            av_loss += policy_loss.item()
            # av_alpha_loss += alpha_loss.item()
        self.iter_count += 1
        # Write new values for tensorboard
        writer.add_scalar("loss", policy_loss.item(), self.iter_count)
        writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        writer.add_scalar("Max. Q", max_Q, self.iter_count)
        
        writer.add_scalar('loss/critic_1', qf1_loss.item(), self.iter_count)
        writer.add_scalar('loss/critic_2', qf2_loss.item(), self.iter_count)
        writer.add_scalar('loss/avg_entropy_loss', av_alpha_loss/iterations, self.iter_count)

        writer.add_scalar('entropy/avg_entropy', av_entropy/iterations, self.iter_count)
        writer.add_scalar('entropy/avg_alpha', av_alpha/iterations, self.iter_count)
        writer.add_scalar('entropy/avg_td_error', av_td_error/iterations, self.iter_count)
        writer.add_scalar('entropy/avg_delta_distance', av_delta_distance/iterations, self.iter_count)

        # writer.add_scalar('entropy_temprature/alpha', alpha_tlogs.item(), self.iter_count)

    def save(self, filename, directory):
        # 保存 actor, critic, 和 temperature 参数
        torch.save(self.policy.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"{directory}/{filename}_critic_target.pth")
        torch.save(self.critic_optim.state_dict(), f"{directory}/{filename}_critic_optim.pth")
        torch.save(self.policy_optim.state_dict(), f"{directory}/{filename}_policy_optim.pth")


    def load(self, filename, directory):
        # 加载 actor, critic, 和 temperature 参数
        self.policy.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.critic_target.load_state_dict(torch.load(f"{directory}/{filename}_critic_target.pth"))
        self.critic_optim.load_state_dict(torch.load(f"{directory}/{filename}_critic_optim.pth"))
        self.policy_optim.load_state_dict(torch.load(f"{directory}/{filename}_policy_optim.pth"))



# Create the network storage folders
if not os.path.exists("./results"):
    os.makedirs("./results")
if save_model and not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorch_models")
if save_model and not os.path.exists("./models"):
    os.makedirs("./models")

# Create the training environment
environment_dim = 20
robot_dim = 4
env = GazeboEnv("multi_robot_scenario.launch", environment_dim)
time.sleep(5)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

state_dim = environment_dim + robot_dim
action_dim = 2
max_action = 1


# Create the network
network = SAC(state_dim, action_dim, alpha, hidden_dim, lr)
# Create a replay buffer
replay_buffer = ReplayBuffer(buffer_size, seed)
if load_model:
    try:
        network.load(file_name, "./pytorch_models")
    except:
        print(
            "Could not load the stored model parameters, initializing training with random parameters"
        )

# Create evaluation data store
evaluations = []

timestep = 0
timesteps_since_eval = 0
episode_num = 0
done = True
epoch = 1
episode_reward = 0
episode_timesteps = 0
target = False
total_reward = 0
target_reached = 0
epoch_success = 0
epoch_reward = 0
col_total = 0

count_rand_actions = 0
random_action = []
state_sequnce = []
angle_squence = []
odom_sequnce = []

predictor = GoalPredictor(api_base, api_key, deployment_name, api_version)

# Begin the training loop
while timestep < max_timesteps:

    progress = timestep / max_timesteps
    # On termination of episode
    if done:
        if timestep >2:
            if use_PPER:
                replay_buffer.add_trajectory(traj)
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                progress,
            )

        if (timesteps_since_eval >= eval_freq) and (timestep >= start_steps):
            print("Validating")
            timesteps_since_eval %= eval_freq
            evaluations.append(
                evaluate(network=network, epoch=epoch, eval_episodes=eval_ep)
            )
            network.save(file_name, directory="./pytorch_models")
            np.save("./results/%s" % (file_name), evaluations)
            epoch += 1

        if episode_num != 0:
            # 在 episode 结束时
            target_reached += int(target)
            epoch_success += int(target)
            epoch_reward += episode_reward
            total_reward += episode_reward
            writer.add_scalar('train/avg_reward', total_reward/(episode_num+1), timestep)
            writer.add_scalar('train/steps', episode_timesteps, episode_num)
            writer.add_scalar('train/success', target_reached/(episode_num+1), episode_num)
            writer.add_scalar('train/collision', col_total/(episode_num+1), episode_num)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, done:{}".format(episode_num, timestep, episode_timesteps, round(episode_reward, 2), int(target)))

            sample_rate = 1 - (episode_num/curriculum_steps)
            # print('odom:', odom_sequnce)
            
            # if not target and episode_timesteps > 50 and np.random.uniform(0, 1) < sample_rate and use_LLM_HER:
            if not target and episode_timesteps > 50 and episode_num < curriculum_steps and use_LLM_HER:
                # with open('data.txt', 'w') as file:
                #     np.savetxt(file, state_sequnce, fmt='%f')
                # print('goal:', info['goal_x'], info['goal_y'])
                # if len(state_sequnce) > 50:
                #     state_sequnce = state_sequnce[-50:]
                #     odom_sequnce = odom_sequnce[-50:]
                
                if len(odom_state_sequence) > 500:
                    odom_state_sequence = odom_state_sequence[-200:]

                if info['collision']:
                    reason = 'collision. The collision point is ({}, {})'.format(info['collision_point'][0], info['collision_point'][1])
                elif episode_timesteps > 499:
                    reason = 'timeout'
                
                # new_state, new_goal_x, new_goal_y = predictor.get_new_goal(info['goal_x'], info['goal_y'], reason, state_sequnce, odom_sequnce, angle_squence)
                # new_state_sequence, new_next_state_sequence, new_odom_x_sequence, new_odom_y_sequence, new_angle_sequence, new_goal_x, new_goal_y = predictor.get_new_goal(info['goal_x'], info['goal_y'], reason, state_sequnce, odom_sequnce, angle_squence)
                new_goal_x, new_goal_y = predictor.get_new_goal(info['goal_x'], info['goal_y'], reason, odom_state_sequence)
                # new_state_sequence, new_next_state_sequence, new_odom_x_sequence, new_odom_y_sequence, new_angle_sequence, new_goal_x, new_goal_y = result
                # start_index, end_index = predictor.get_index(odom_state_sequence)
                if new_goal_x != info['goal_x'] or new_goal_y != info['goal_y']:
                # if end_index-start_index > 9:
                #     env.publish_traj(odom_state_sequence[start_index:end_index+1])
                #     # env.perturb_goal(new_goal_x, new_goal_y, state[:-4])
                #     new_goal_x = odom_state_sequence[end_index][0]
                #     new_goal_y = odom_state_sequence[end_index][1]
                    env.publish_last_old_markers(info['goal_x'], info['goal_y'])
                    env.publish_last_new_markers(new_goal_x, new_goal_y)
                    print('new goal generated.')
                    distance = np.sqrt((new_goal_x - info['goal_x'])**2 + (new_goal_y - info['goal_y'])**2)
                    reward_scale = np.clip(1 - distance/10, 0.5, 1)
                    writer.add_scalar('train/goal_distance', distance, episode_num)
                    init_state, _, _, _, _, _, init_odom_x, init_odom_y, init_angel, _, _, _ = replay_buffer.get_last_episode(0)
                    new_state, _, _ = env.calculate_reward_and_state(init_state, init_odom_x, init_odom_y, init_angel, new_goal_x, new_goal_y)
                    if use_PPER:
                        traj = Trajectory(new_state)
                    for i in range(1, episode_timesteps):
                        # 获取当前 episode 的 transition
                        # print('start:', replay_buffer.episode_start_indices)
                        # print('i:', i)
                        old_next_state, old_action, old_reward, old_done_bool, old_done, old_next_state, odom_x, odom_y, angle, goal_x, goal_y, priority = replay_buffer.get_last_episode(i)
                        # print('length:', replay_buffer.size(), 'steps:', episode_timesteps)

                        # 重新计算 reward 和 state
                        # new_state, new_reward, new_target = env.calculate_reward_and_state(new_state_sequence[i], new_odom_x_sequence[i], new_odom_y_sequence[i], new_angle_sequence[i], new_goal_x, new_goal_y)
                        new_next_state, new_reward, new_target = env.calculate_reward_and_state(old_next_state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                        #new_next_state, _, _ = env.calculate_reward_and_state(old_next_state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                        # new_next_state = new_next_state_sequence[i]
                        # print('new_next_state:', np.shape(next_state))

                        new_done = 1 if new_target else int(old_done)
                        new_done_bool = 1 if new_target else int(old_done_bool)

                        # 更新 replay buffer 中的 transition
                        # replay_buffer.update_last_episode(i, new_state, old_action, new_reward, new_done_bool, new_done, new_next_state, odom_x, odom_y, angle, goal_x, goal_y)

                        

                        if use_PPER:
                            traj.add(new_next_state, old_action, new_reward, new_done_bool, new_done, new_next_state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                        elif use_TDPER:
                            max_priority = replay_buffer._get_max_priority()
                            replay_buffer.add(new_state, old_action, new_reward, new_done_bool, new_done, new_next_state, odom_x, odom_y, angle, new_goal_x, new_goal_y, max_priority)
                        else:
                            replay_buffer.add(new_state, old_action, new_reward, new_done_bool, new_done, new_next_state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                        new_state = new_next_state
                        if new_done:
                            if use_PPER:
                                replay_buffer.add_trajectory()
                            break

        
        state_sequnce = []
        odom_sequnce = []
        angle_squence = []
        odom_state_sequence = []
        state, info = env.reset()

        if use_PPER:
            traj = Trajectory(state)
        # print('shape:', np.shape(state))
        #print(state)
        done = False
        # state_sequnce.append(state)
        # odom_state_sequence.append(info['odom_state'])

        # if episode_num % 100 == 0:
        #     writer.add_scalar('train/epoch_success', epoch_success/100, episode_num/100)
        #     epoch_success = 0

        if episode_num % 10 == 0:
            writer.add_scalar('train/epoch_success', epoch_success/10, episode_num/10)
            writer.add_scalar('train/epoch_reward', epoch_reward/10, episode_num/10)
            epoch_success = 0
            epoch_reward = 0

        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

    # add some exploration noise
    if expl_noise > expl_min:
        expl_noise = expl_noise - ((1 - expl_min) / expl_decay_steps)

    action = network.get_action(np.array(state))
    action = (action + np.random.normal(0, expl_noise, size=action_dim)).clip(
        -max_action, max_action
    )

    # If the robot is facing an obstacle, randomly force it to take a consistent random action.
    # This is done to increase exploration in situations near obstacles.
    # Training can also be performed without it
    if random_near_obstacle:
        if (
            np.random.uniform(0, 1) > 0.85
            and min(state[4:-8]) < 0.6
            and count_rand_actions < 1
        ):
            count_rand_actions = np.random.randint(8, 15)
            random_action = np.random.uniform(-1, 1, 2)

        if count_rand_actions > 0:
            count_rand_actions -= 1
            action = random_action
            action[0] = -1

    # Update action to fall in range [0,1] for linear velocity and [-1,1] for angular velocity
    a_in = [(action[0] + 1) / 2, action[1]]
    next_state, reward, done, target, info = env.step(a_in)
    

    done_bool = 0 if episode_timesteps + 1 == max_ep else int(done)
    done = 1 if episode_timesteps + 1 == max_ep else int(done)
    episode_reward += reward
    if info['collision']:
        col_total += 1
        collision = True
    else:
        collision = False

    # Save the tuple in replay buffer
    if use_TDPER:
        max_priority = replay_buffer._get_max_priority()
        replay_buffer.add(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'], info['angle'], info["goal_x"], info["goal_y"], max_priority)
    elif use_PPER:
        traj.store_step(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'], info['angle'], info["goal_x"], info["goal_y"])
    elif use_PQPER:
        max_priority = replay_buffer._get_max_priority()
        replay_buffer.add(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'] , info['angle'], info["goal_x"], info["goal_y"], max_priority)
    else:
        replay_buffer.add(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'] , info['angle'], info["goal_x"], info["goal_y"])

    # Update the counters
    state = next_state
    state_sequnce.append(state.flatten().tolist())
    angle_squence.append(info['angle'])
    odom_sequnce.append(np.round(np.array([float(info['odom_x']), float(info['odom_y'])]),4))
    odom_state_sequence.append(info['odom_state'])
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
