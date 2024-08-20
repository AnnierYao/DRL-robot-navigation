import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from numpy import inf
from torch.utils.tensorboard import SummaryWriter

from replay_buffer import ReplayBuffer
from velodyne_env import GazeboEnv
from gpt_feedback import GoalPredictor

writer = SummaryWriter()

api_base = "https://eastusengine.openai.azure.com/"
api_key = '9ae06f2709144320b0b4645c587a3492'
deployment_name = 'gpt-4o-mini'
api_version = '2023-03-15-preview'

def evaluate(network, epoch, eval_episodes=10):
    avg_reward = 0.0
    col = 0
    for _ in range(eval_episodes):
        count = 0
        state, info = env.reset()
        done = False
        while not done and count < 501:
            action = network.get_action(np.array(state))
            a_in = [(action[0] + 1) / 2, action[1]]
            state, reward, done, _, info = env.step(a_in)
            avg_reward += reward
            count += 1
            if reward < -90:
                col += 1
    avg_reward /= eval_episodes
    avg_col = col / eval_episodes
    print("..............................................")
    print(
        "Average Reward over %i Evaluation Episodes, Epoch %i: %f, %f"
        % (eval_episodes, epoch, avg_reward, avg_col)
    )
    print("..............................................")
    writer.add_scalar('test/reward', avg_reward, epoch)
    writer.add_scalar('test/collision', avg_col, epoch) 
    return avg_reward



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.mean_layer = nn.Linear(600, action_dim)
        self.log_std_layer = nn.Linear(600, action_dim)
        self.max_action = max_action

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        mean = self.mean_layer(s)
        log_std = self.log_std_layer(s).clamp(-20, 2)  # 将标准差的对数值限制在一定范围内
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action = dist.rsample()  # 通过重参数技巧进行采样
        action = torch.tanh(action) * self.max_action  # 使用 tanh 来限制动作范围
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算动作的 log 概率
        log_prob -= (2 * (torch.log(torch.tensor(2.0)) - action - F.softplus(-2 * action))).sum(dim=-1)  # 修正 tanh 变换的 log 概率
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.layer_1 = nn.Linear(state_dim + action_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, 1)

        # Q2 architecture (similar to TD3)
        self.layer_4 = nn.Linear(state_dim + action_dim, 800)
        self.layer_5 = nn.Linear(800, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)

        # Q1 forward pass
        q1 = F.relu(self.layer_1(sa))
        q1 = F.relu(self.layer_2(q1))
        q1 = self.layer_3(q1)

        # Q2 forward pass
        q2 = F.relu(self.layer_4(sa))
        q2 = F.relu(self.layer_5(q2))
        q2 = self.layer_6(q2)

        return q1, q2



class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, alpha=0.2, target_entropy=-1.0):
        # Actor and Critic networks
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.alpha = alpha
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.target_entropy = target_entropy

        self.max_action = max_action
        self.iter_count = 0

    def get_action(self, state):
        state = torch.Tensor(state).to(device)
        action, _ = self.actor(state)

        return action.detach().cpu().numpy()


    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2,  noise_clip=0.5):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
        av_alpha_loss = 0
        for it in range(iterations):
            # Sample a batch from the replay buffer
            (
                batch_states,
                batch_actions,
                batch_rewards,
                batch_dones,
                batch_next_states,
            ) = replay_buffer.sample_batch(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            # Update Critic
            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_state)
                # Add noise to the action
                # noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
                # noise = noise.clamp(-noise_clip, noise_clip)
                # next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                target_Q1, target_Q2 = self.critic_target(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
                av_Q += torch.mean(target_Q)
                max_Q = max(max_Q, torch.max(target_Q))
                target_Q = reward + (1 - done) * discount * target_Q

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update Actor
            action, log_prob = self.actor(state)
            Q1, Q2 = self.critic(state, action)
            actor_loss = (self.alpha * log_prob - torch.min(Q1, Q2)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update alpha
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # Update Critic target networks with soft update
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            av_loss += actor_loss
            av_alpha_loss += alpha_loss
        self.iter_count += 1
        # Write new values for tensorboard
        writer.add_scalar("alpha", av_alpha_loss/iterations, self.iter_count)
        writer.add_scalar("loss", av_loss / iterations, self.iter_count)
        writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        writer.add_scalar("Max. Q", max_Q, self.iter_count)

    def save(self, filename, directory):
        # 保存 actor, critic, 和 temperature 参数
        torch.save(self.actor.state_dict(), f"{directory}/{filename}_actor.pth")
        torch.save(self.critic.state_dict(), f"{directory}/{filename}_critic.pth")
        torch.save(self.log_alpha, f"{directory}/{filename}_temperature.pth")

    def load(self, filename, directory):
        # 加载 actor, critic, 和 temperature 参数
        self.actor.load_state_dict(torch.load(f"{directory}/{filename}_actor.pth"))
        self.critic.load_state_dict(torch.load(f"{directory}/{filename}_critic.pth"))
        self.log_alpha = torch.load(f"{directory}/{filename}_temperature.pth")


# Set the parameters for the implementation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # cuda or cpu
# device = torch.device("cpu")
seed = 1  # Random seed number
eval_freq = 5e3  # After how many steps to perform the evaluation
start_steps = 1e4  # Number of steps to perform the evaluation
max_ep = 500  # maximum number of steps per episode
eval_ep = 10  # number of episodes for evaluation
max_timesteps = 2e6  # Maximum number of steps to perform
expl_noise = 1  # Initial exploration noise starting value in range [expl_min ... 1]
expl_decay_steps = (
    500000  # Number of steps over which the initial exploration noise will decay over
)
expl_min = 0.1  # Exploration noise after the decay in range [0...expl_noise]
batch_size = 40  # Size of the mini-batch
discount = 0.99999  # Discount factor to calculate the discounted future reward (should be close to 1)
tau = 0.005  # Soft target update variable (should be close to 0)
policy_noise = 0.2  # Added noise for exploration
noise_clip = 0.5  # Maximum clamping values of the noise
policy_freq = 2  # Frequency of Actor network updates
buffer_size = 1e6  # Maximum size of the buffer
file_name = "TD3_velodyne_sac"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
load_model = False  # Weather to load a stored model
random_near_obstacle = True  # To take random actions near obstacles or not
use_LLM_HER = False  # Weather to use LLM HER or not
use_HER = False  # Weather to use HER or not

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
network = SAC(state_dim, action_dim, max_action)
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
col_total = 0

count_rand_actions = 0
random_action = []
state_sequnce = []
odom_sequnce = []

predictor = GoalPredictor(api_base, api_key, deployment_name, api_version)

# Begin the training loop
while timestep < max_timesteps:

    # On termination of episode
    if done:
        if timestep != 0:
            network.train(
                replay_buffer,
                episode_timesteps,
                batch_size,
                discount,
                tau,
                policy_noise,
                noise_clip,
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
            total_reward += episode_reward
            writer.add_scalar('train/avg_reward', total_reward/(episode_num+1), timestep)
            writer.add_scalar('train/steps', episode_timesteps, episode_num)
            writer.add_scalar('train/success', target_reached/(episode_num+1), episode_num)
            writer.add_scalar('train/collision', col_total/(episode_num+1), episode_num)
            print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, done:{}".format(episode_num, timestep, episode_timesteps, round(episode_reward, 2), int(target)))

            if not target and episode_timesteps>1 and np.random.uniform(0, 1) < 0.5 and episode_num < 200 and use_LLM_HER:
                # with open('data.txt', 'w') as file:
                #     np.savetxt(file, state_sequnce, fmt='%f')
                # print('goal:', info['goal_x'], info['goal_y'])
                # if len(state_sequnce) > 50:
                    # state_sequnce = state_sequnce[-50:]
                    # odom_sequnce = odom_sequnce[-50:]
                if len(odom_state_sequence) > 50:
                    odom_state_sequence = odom_state_sequence[-50:]

                if info['collision']:
                    reason = 'collision. The collision point is ({}, {})'.format(info['collision_point'][0], info['collision_point'][1])
                elif episode_timesteps > 499:
                    reason = 'timeout'
                
                # new_goal_x, new_goal_y = predictor.get_new_goal(info['goal_x'], info['goal_y'], state_sequnce, odom_sequnce)
                new_goal_x, new_goal_y = predictor.get_new_goal(info['goal_x'], info['goal_y'], reason, odom_state_sequence)
                # print('new goal:', new_goal_x, new_goal_y)
                # print('old goal:', info['goal_x'], info['goal_y'])
                # print(new_goal_x != info['goal_x'], new_goal_y != info['goal_y'])

                if new_goal_x != info['goal_x'] or new_goal_y != info['goal_y']:
                    env.perturb_goal(new_goal_x, new_goal_y, state[:-4])
                    env.publish_last_old_markers(info['goal_x'], info['goal_y'])
                    env.publish_last_new_markers(new_goal_x, new_goal_y)
                    print('new goal generated.')
                    for i in range(episode_timesteps):
                        # 获取当前 episode 的 transition
                        # print('start:', replay_buffer.episode_start_indices)
                        # print('i:', i)
                        old_state, old_action, old_reward, old_done_bool, old_done, old_next_state, odom_x, odom_y, angle, goal_x, goal_y = replay_buffer.get_last_episode(i)
                        # print('length:', replay_buffer.size(), 'steps:', episode_timesteps)

                        # 重新计算 reward 和 state
                        new_state, new_reward, new_target = env.calculate_reward_and_state(old_state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                        new_next_state, _, _ = env.calculate_reward_and_state(old_next_state, odom_x, odom_y, angle, new_goal_x, new_goal_y)
                        # print('new_next_state:', np.shape(next_state))

                        new_done = 1 if new_target else int(old_done)
                        new_done_bool = 1 if new_target else int(old_done_bool)

                        # 更新 replay buffer 中的 transition
                        replay_buffer.update_last_episode(i, new_state, old_action, new_reward, new_done_bool, new_done, new_next_state)
                        # replay_buffer.add(new_state, old_action, new_reward, new_done_bool, new_done, new_next_state)

        
        # state_sequnce = []
        # odom_sequnce = []
        odom_state_sequence = []
        state, info = env.reset()
        # print('shape:', np.shape(state))
        #print(state)
        done = False
        # state_sequnce.append(state)
        odom_state_sequence.append(info['odom_state'])

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
    if reward < -90:
        col_total += 1
        collision = True
    else:
        collision = False

    # Save the tuple in replay buffer
    replay_buffer.add(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'] , info['angle'], info["goal_x"], info["goal_y"])

    # Update the counters
    state = next_state
    # state_sequnce.append(state.flatten().tolist())
    # odom_sequnce.append(np.round(np.array([float(info['odom_x']), float(info['odom_y'])]),4))
    odom_state_sequence.append(info['odom_state'])
    episode_timesteps += 1
    timestep += 1
    timesteps_since_eval += 1

# After the training is done, evaluate the network and save it
evaluations.append(evaluate(network=network, epoch=epoch, eval_episodes=eval_ep))
if save_model:
    network.save("%s" % file_name, directory="./models")
np.save("./results/%s" % file_name, evaluations)
