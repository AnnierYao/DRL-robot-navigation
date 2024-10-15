import os
import time

import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
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
max_timesteps = 5e5  # Maximum number of steps to perform
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
name = "TD3_proportional_pathquality"  # name of the file to store the policy
save_model = True  # Weather to save the model or not
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
use_PQPER = True  # Whether to use PQPER or not
use_PPER = False  # Whether to use PPER or not
# from replay_buffer import ReplayBuffer
# from replay_buffer_PER import ReplayBuffer
# from replay_buffer_her import ReplayBuffer
# from replay_buffer_PQPER import ReplayBuffer, Trajectory
# from replay_buffer_PPER import ReplayBuffer
from replay_buffer_PQPER import ReplayBuffer
lr = 0.001  # Learning rate for the networks
alpha = 0.2  # Alpha value for the SAC algorithm
hidden_dim = 500


file_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(name, reward_type, feedback_form if use_LLM_HER else "", env_name,batch_size,lr, hidden_dim,
                                                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 
                                                    "HER" if use_HER else "", 
                                                    "LLM_HER" if use_LLM_HER else "", curriculum_steps if use_LLM_HER else "","PER_HER" if use_TDPER else "", "PQPER" if use_PQPER else "", "PPER" if use_PPER else "")
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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2 = nn.Linear(800, 600)
        self.layer_3 = nn.Linear(600, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, s):
        s = F.relu(self.layer_1(s))
        s = F.relu(self.layer_2(s))
        a = self.tanh(self.layer_3(s))
        return a

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.layer_1 = nn.Linear(state_dim, 800)
        self.layer_2_s = nn.Linear(800, 600)
        self.layer_2_a = nn.Linear(action_dim, 600)
        self.layer_3 = nn.Linear(600, 1)

        self.layer_4 = nn.Linear(state_dim, 800)
        self.layer_5_s = nn.Linear(800, 600)
        self.layer_5_a = nn.Linear(action_dim, 600)
        self.layer_6 = nn.Linear(600, 1)

    def forward(self, s, a):
        s1 = F.relu(self.layer_1(s))
        self.layer_2_s(s1)
        self.layer_2_a(a)
        s11 = torch.mm(s1, self.layer_2_s.weight.data.t())
        s12 = torch.mm(a, self.layer_2_a.weight.data.t())
        s1 = F.relu(s11 + s12 + self.layer_2_a.bias.data)
        q1 = self.layer_3(s1)

        s2 = F.relu(self.layer_4(s))
        self.layer_5_s(s2)
        self.layer_5_a(a)
        s21 = torch.mm(s2, self.layer_5_s.weight.data.t())
        s22 = torch.mm(a, self.layer_5_a.weight.data.t())
        s2 = F.relu(s21 + s22 + self.layer_5_a.bias.data)
        q2 = self.layer_6(s2)
        return q1, q2




class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        # Initialize the Actor network
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())

        # Initialize the Critic networks
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())

        self.max_action = max_action
        # self.writer = SummaryWriter()
        self.iter_count = 0

    def get_action(self, state):
        # Function to get the action from the actor
        state = torch.Tensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    # training cycle
    def train(
        self,
        replay_buffer,
        iterations,
        batch_size=100,
        discount=1,
        tau=0.005,
        policy_noise=0.2,  # discount=0.99
        noise_clip=0.5,
        policy_freq=2,
    ):
        av_Q = 0
        max_Q = -inf
        av_loss = 0
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

            next_action = self.actor_target(next_state)

            # Add noise to the action
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Calculate the Q values from the critic-target network for the next state-action pair
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)

            # Select the minimal Q value from the 2 calculated values
            target_Q = torch.min(target_Q1, target_Q2)
            av_Q += torch.mean(target_Q)
            max_Q = max(max_Q, torch.max(target_Q))
            # Calculate the final Q value from the target network parameters by using Bellman equation
            target_Q = reward + ((1 - done) * discount * target_Q).detach()

            # Get the Q values of the basis networks with the current parameters
            current_Q1, current_Q2 = self.critic(state, action)

            # Calculate the loss between the current Q value and the target Q value
            # loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            td_error_qf1 = current_Q1 - target_Q
            td_error_qf2 = current_Q2 - target_Q

            # 计算加权平方误差
            weighted_squared_td_error1 = is_weights * (td_error_qf1 ** 2)
            weighted_squared_td_error2 = is_weights * (td_error_qf2 ** 2)

            # 计算加权 MSE 损失
            qf1_loss = weighted_squared_td_error1.mean()
            qf2_loss = weighted_squared_td_error2.mean()
            # qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            # qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            loss = qf1_loss + qf2_loss

            # Perform the gradient descent
            self.critic_optimizer.zero_grad()
            loss.backward()
            self.critic_optimizer.step()

            if it % policy_freq == 0:
                # Maximize the actor output value by performing gradient descent on negative Q values
                # (essentially perform gradient ascent)
                actor_grad, _ = self.critic(state, self.actor(state))
                actor_grad = -actor_grad.mean()
                self.actor_optimizer.zero_grad()
                actor_grad.backward()
                self.actor_optimizer.step()

                # Use soft update to update the actor-target network parameters by
                # infusing small amount of current parameters
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
                # Use soft update to update the critic-target network parameters by infusing
                # small amount of current parameters
                for param, target_param in zip(
                    self.critic.parameters(), self.critic_target.parameters()
                ):
                    target_param.data.copy_(
                        tau * param.data + (1 - tau) * target_param.data
                    )
            av_loss += loss.item()
        self.iter_count += 1
        # Write new values for tensorboard
        writer.add_scalar("loss", loss.item(), self.iter_count)
        writer.add_scalar("Av. Q", av_Q / iterations, self.iter_count)
        writer.add_scalar("Max. Q", max_Q, self.iter_count)


    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, filename))
        torch.save(self.critic.state_dict(), "%s/%s_critic.pth" % (directory, filename))

    def load(self, filename, directory):
        self.actor.load_state_dict(
            torch.load("%s/%s_actor.pth" % (directory, filename))
        )
        self.critic.load_state_dict(
            torch.load("%s/%s_critic.pth" % (directory, filename))
        )



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
network = TD3(state_dim, action_dim, max_action)
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
        if timestep != 0:
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
                        elif use_PER_HER:
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
    if use_PER_HER:
        max_priority = replay_buffer._get_max_priority()
        replay_buffer.add(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'], info['angle'], info["goal_x"], info["goal_y"], max_priority)
    elif use_PPER:
        traj.store_step(state, action, reward, done_bool, done, next_state, info['odom_x'], info['odom_y'], info['angle'], info["goal_x"], info["goal_y"])
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
