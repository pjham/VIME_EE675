import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import random
from collections import deque
import numpy as np
import gym
import math
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import replay_buffer
from vime import vime
import pickle
import os

class normallized_action_wrapper(gym.ActionWrapper):
    # * because the tanh value range is [-1, 1], so change the env action range
    def _action(self, action):
        # * change action range from [-1, 1] to [env.low, env.high]
        low = self.action_space.low
        high = self.action_space.high

        action = (action + 1) / 2 * (high - low) - 2
        action = np.clip(action, low, high)
        return action

    def _reverse_action(self):
        # * change action range from [env.low, env.high] to [-1, 1]
        low = self.action_space.low
        high = self.action_space.high

        action = (action - low) / ((high - low) / 2) - 1
        action = np.clip(action, -1, 1)
        return action


class policy_net(nn.Module):
    # * SAC trains a stochastic policy, not a deterministic policy which like TD3 and DDPG
    def __init__(self, input_dim, output_dim, min_log_sigma=-20., max_log_sigma=2.):
        super(policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc_mu = nn.Linear(128, self.output_dim)
        self.fc_sigma = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        mu = self.fc_mu(x)
        # * standard deviations are parameterized, the way not same as VPG, PPO and TRPO
        log_sigma = self.fc_sigma(x)
        log_sigma = torch.clamp(log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma

    def act(self, input):
        mu, log_sigma = self.forward(input)
        sigma = torch.exp(log_sigma)
        dist = Normal(mu, sigma)
        # * reparameterization trick: recognize the difference of sample() and rsample()
        action = dist.rsample()
        tanh_action = torch.tanh(action)
        # * the log-probabilities of actions can be calculated in closed forms
        log_prob = dist.log_prob(action)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2)).sum(1, keepdim=True)
        return tanh_action, log_prob


class value_net(nn.Module):
    def __init__(self, input1_dim, input2_dim, output_dim):
        super(value_net, self).__init__()
        self.input1_dim = input1_dim
        self.input2_dim = input2_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input1_dim + self.input2_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input1, input2):
        x = torch.cat([input1, input2], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class sac_vime(object):
    def __init__(self, env, batch_size, learning_rate, exploration, episode, gamma, alpha, capacity, rho, update_iter, update_every, render, log, vime_model=None):
        self.env = env
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.exploration = exploration
        self.episode = episode
        self.gamma = gamma
        self.alpha = alpha
        self.capacity = capacity
        self.rho = rho
        self.update_iter = update_iter
        self.update_every = update_every
        self.render = render
        self.log = log
        self.vime_model = vime_model

        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.value_net1 = value_net(self.observation_dim, self.action_dim, 1)
        self.value_net2 = value_net(self.observation_dim, self.action_dim, 1)
        self.target_value_net1 = value_net(self.observation_dim, self.action_dim, 1)
        self.target_value_net2 = value_net(self.observation_dim, self.action_dim, 1)
        self.policy_net = policy_net(self.observation_dim, self.action_dim)
        self.target_value_net1.load_state_dict(self.value_net1.state_dict())
        self.target_value_net2.load_state_dict(self.value_net2.state_dict())

        self.buffer = replay_buffer(capacity=self.capacity)
        if self.vime_model is not None:
            self.vime_traj_buffer = []

        self.value_optimizer1 = torch.optim.Adam(self.value_net1.parameters(), lr=self.learning_rate)
        self.value_optimizer2 = torch.optim.Adam(self.value_net2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)

        self.weight_reward = None
        self.count = 0
        self.train_count = 0
        self.writer = SummaryWriter('runs/sac_vime')
        self.train_rewards = {}
        self.train_steps = {}

    def soft_update(self):
        for param, target_param in zip(self.value_net1.parameters(), self.target_value_net1.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)
        for param, target_param in zip(self.value_net2.parameters(), self.target_value_net2.parameters()):
            target_param.detach().copy_(param.detach() * (1 - self.rho) + target_param.detach() * self.rho)

    def train(self):
        observation, action, reward, next_observation, done = self.buffer.sample(self.batch_size)

        observation = torch.FloatTensor(observation)
        action = torch.FloatTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_observation = torch.FloatTensor(next_observation)
        done = torch.FloatTensor(done).unsqueeze(1)

        value_loss1_buffer = []
        value_loss2_buffer = []
        policy_loss_buffer = []
        for _ in range(self.update_iter):
            next_action, log_prob = self.policy_net.act(next_observation)
            target_q_value1 = self.target_value_net1.forward(next_observation, next_action)
            target_q_value2 = self.target_value_net2.forward(next_observation, next_action)
            target_q = reward + (1 - done) * self.gamma * (torch.min(target_q_value1, target_q_value2) - self.alpha * log_prob)
            target_q = target_q.detach()

            q1 = self.value_net1.forward(observation, action)
            q2 = self.value_net2.forward(observation, action)
            value_loss1 = (q1 - target_q).pow(2).mean()
            value_loss2 = (q2 - target_q).pow(2).mean()
            value_loss1_buffer.append(value_loss1.detach().item())
            value_loss2_buffer.append(value_loss2.detach().item())

            self.value_optimizer1.zero_grad()
            value_loss1.backward()
            nn.utils.clip_grad_norm_(self.value_net1.parameters(), 0.5)
            self.value_optimizer1.step()

            self.value_optimizer2.zero_grad()
            value_loss2.backward()
            nn.utils.clip_grad_norm_(self.value_net2.parameters(), 0.5)
            self.value_optimizer2.step()

            sample_action, sample_log_prob = self.policy_net.act(observation)
            sample_q1 = self.value_net1.forward(observation, sample_action)
            sample_q2 = self.value_net2.forward(observation, sample_action)
            policy_loss = - (torch.min(sample_q1, sample_q2) - self.alpha * sample_log_prob)
            policy_loss = policy_loss.mean()
            policy_loss_buffer.append(policy_loss.detach().item())

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.policy_optimizer.step()

            self.soft_update()

        if self.vime_model is not None:
            elbo = self.vime_model.update(self.buffer)

        if self.log:
            self.writer.add_scalar('value_loss1', np.mean(value_loss1_buffer), self.train_count)
            self.writer.add_scalar('value_loss2', np.mean(value_loss2_buffer), self.train_count)
            self.writer.add_scalar('policy_loss', np.mean(policy_loss_buffer), self.train_count)
            if self.vime_model is not None:
                self.writer.add_scalar('elbo', elbo, self.train_count)

    def vime_process(self):
        info_gains = np.array([self.vime_model.calc_info_gain(obs, action, next_obs) for (obs, action, _, next_obs, _) in self.vime_traj_buffer])
        rewards_org = np.array([reward for (_, _, reward, _, _) in self.vime_traj_buffer])
        rewards = self.vime_model.calc_curiosity_reward(rewards_org, info_gains)
        self.vime_model.store_kl(info_gains)
        for (obs, action, _, next_obs, done), reward in zip(self.vime_traj_buffer, rewards):
            self.buffer.store(obs, action, reward, next_obs, done)
        del self.vime_traj_buffer[:]

    def run(self):
        print("run function called")
        recent_rewards = deque(maxlen=10)
        for i in range(self.episode):
            obs = self.env.reset()
            total_reward = 0
            if self.render:
                self.env.render()
            while True:
                if i >= self.exploration:
                    tanh_action, _ = self.policy_net.act(torch.FloatTensor(np.expand_dims(obs, 0)))
                    action = tanh_action.detach().item()
                else:
                    action = np.random.uniform(-1., 1.)
                next_obs, reward, done, _ = self.env.step(action)
                if self.render:
                    self.env.render()

                if self.vime_model is not None:
                    self.vime_traj_buffer.append([obs, action, reward, next_obs, done])
                else:
                    self.buffer.store(obs, action, reward, next_obs, done)

                self.count += 1
                total_reward += reward
                obs = next_obs

                if (self.count % self.update_every) == 0 and i >= self.exploration:
                    self.train_count += 1
                    if self.vime_model is not None:
                        self.vime_process()
                    self.train()
                if done:
                    if not self.weight_reward:
                        self.weight_reward = total_reward
                    else:
                        self.weight_reward = self.weight_reward * 0.99 + total_reward * 0.01
                    recent_rewards.append(total_reward)
                    self.train_rewards[i] = total_reward
                    self.train_steps[i] = self.count
                    
                    if self.log:
                        self.writer.add_scalar('reward', total_reward, i + 1)
                        self.writer.add_scalar('weight_reward', self.weight_reward, i + 1)
                    if (i + 1) % 10 == 0:
                        avg_reward = sum(recent_rewards) / len(recent_rewards)
                        print(f'Episodes {i + 1 - 9} to {i + 1}: Average reward = {avg_reward:.2f}')
                    break
        if self.vime_model is None:
            model_dir = 'results/sac'
        else:
            model_dir = 'results/sac_vime'
        pickle_file = model_dir + '/train_data.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump({'train_rewards': self.train_rewards, 'train_steps': self.train_steps}, f)
        
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.policy_net.state_dict(), os.path.join(model_dir, 'policy_net.pth'))

