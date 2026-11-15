# Create a policy that encapsulates an actor network

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import networks as net
from src.utils import features as utils
import copy
import numpy as np

class NaiveLookaheadPolicy(nn.Module):
    def __init__(self, env, state_dim, action_dim, actor, transition_model, reward_fn, perturbation=0.2, root_degree=5, depth=5):
        super(NaiveLookaheadPolicy, self).__init__()
        # self.actor = net.FeedforwardNeuralNetModel(state_dim, 400, action_dim)
        self.actor = actor
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.perturbation = perturbation
        self.root_degree = root_degree
        self.depth = depth

    def forward(self, state):
        # Instead of just returning the actor's output
        # we simulate the transition model and reward model for a few time steps
        # and choose action that gives best expected return

        # Simulate the transition model and reward model for a few time steps
        state = state.unsqueeze(0)
        actions = self.actor(state).squeeze(0).repeat(self.root_degree, 1) + torch.randn(self.root_degree, self.action_dim) * self.perturbation

        rewards = torch.zeros(self.root_degree)

        for action_idx in range(self.root_degree):
            action = actions[action_idx]
            state = self.transition_model(state, action.unsqueeze(0))
            reward = self.reward_fn(state[0].cpu().detach().numpy(), action.detach().cpu().numpy(), self.env)

            for i in range(self.depth - 1):
                action = self.actor(state)
                state = self.transition_model(state, action)
                reward += self.reward_fn(state.cpu().detach().numpy()[0], action.cpu().detach().numpy()[0], self.env)

            rewards[action_idx] = reward

        best_action_idx = torch.argmax(rewards)

        return actions[best_action_idx]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


# Same as Naive Lookahead Policy but with a stationary policy that only depends on the current observation
class StationaryNaiveLookaheadPolicy(nn.Module):
    def __init__(self, env, obs_dim, event_dim, event_horizon, action_dim, actor, transition_model, reward_fn, perturbation=0.2, root_degree=5, depth=5, num_trajectories=10):
        super(StationaryNaiveLookaheadPolicy, self).__init__()
        # self.actor = net.FeedforwardNeuralNetModel(state_dim, 400, action_dim)
        self.actor = actor
        self.env = env
        self.obs_dim = obs_dim
        self.event_dim = event_dim
        self.event_horizon = event_horizon
        self.state_dim = obs_dim + event_dim * event_horizon
        self.action_dim = action_dim
        self.transition_model = copy.deepcopy(transition_model)

        for param in self.transition_model.parameters():
            param.requires_grad = False

        self.reward_fn = reward_fn
        self.perturbation = perturbation
        self.root_degree = root_degree
        self.depth = depth
        self.num_trajectories = num_trajectories

    def forward(self, state):
        # Instead of just returning the actor's output
        # we simulate the transition model and reward model for a few time steps
        # and choose action that gives best expected return

        # Simulate the transition model and reward model for a few time steps
        obs = state[:self.obs_dim].unsqueeze(0)
        initial_state = state.unsqueeze(0)
        action_perturbation = torch.randn(self.root_degree+1, self.action_dim)
        action_perturbation[0]*=0
        actions = self.actor(obs).squeeze(0).repeat(self.root_degree+1, 1) + action_perturbation * self.perturbation
        initial_states = state.repeat(self.root_degree+1, 1)
        rewards = torch.zeros(self.root_degree+1)
        not_done = np.ones(self.root_degree+1)

        actions = torch.clip(actions, -1.0, 1.0)
        states = self.transition_model(initial_states, actions)
        current_reward = self.reward_fn(states, actions, self.env)
        rewards += current_reward
        not_done = not_done * np.where(np.abs(current_reward)<1e-6, 0.0, 1.0)

        for i in range(self.depth-1):
            actions_1 = self.actor(states[:,:self.obs_dim])
            states = self.transition_model(states, actions_1)
            current_reward = self.reward_fn(states, actions_1, self.env)
            rewards += current_reward*not_done
            not_done = not_done * np.where(np.abs(current_reward)<1e-6, 0.0, 1.0)

        best_action_idx = torch.argmax(rewards)

        return actions[best_action_idx]

        for action_idx in range(self.root_degree+1):
            action = actions[action_idx]
            action = torch.clip(action, -1.0, 1.0)
            state = self.transition_model(initial_state, action.unsqueeze(0))
            reward = self.reward_fn(state[0], action, self.env)

            if abs(reward)<1e-6:
                continue

            for i in range(self.depth - 1):
                action = self.actor(state[:, :self.obs_dim])
                action = torch.clip(action, -1.0, 1.0)
                state = self.transition_model(state, action)
                reward += self.reward_fn(state[0], action[0], self.env)

                if abs(reward)<1e-6:
                    break

            rewards[action_idx] = reward

        best_action_idx = torch.argmax(rewards)

        return actions[best_action_idx]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def update_transition_model(self, state_dict):
        self.transition_model.load_state_dict(state_dict)



# Intelligent look-ahead policy that takes transition and event distributions separately
class IntelligentLookaheadPolicy(nn.Module):
    def __init__(self, env, obs_dim, event_dim, event_horizon, action_dim, actor, event_process, transition_model, reward_fn, perturbation=0.2, root_degree=5, depth=5):
        super(IntelligentLookaheadPolicy, self).__init__()
        # self.actor = net.FeedforwardNeuralNetModel(state_dim, 400, action_dim)
        self.actor = actor
        self.event_process = event_process
        self.env = env
        self.state_dim = obs_dim
        self.event_dim = event_dim
        self.event_horizon = event_horizon
        self.action_dim = action_dim
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.perturbation = perturbation
        self.root_degree = root_degree
        self.depth = depth

    def forward(self, obs, event_history):
        # Instead of just returning the actor's output
        # we simulate the transition model and reward model for a few time steps
        # and choose action that gives best expected return

        # Initialize the event process with the event history till now
        self.event_process.init_history(event_history)


        # Simulate the transition model, event process and reward model for a few time steps
        fixed_event_history = utils.get_fixed_length_vector(event_history, self.event_horizon).flatten()
        state = torch.cat((obs, torch.tensor(fixed_event_history)), dim=0).unsqueeze(0).float()
        actions = self.actor(state).squeeze(0).repeat(self.root_degree, 1) + torch.randn(self.root_degree, self.action_dim) * self.perturbation

        rewards = torch.zeros(self.root_degree)

        for action_idx in range(self.root_degree):
            action = actions[action_idx]
            action=torch.clip(action, -1.0, 1.0)
            obs = self.transition_model(state, action.unsqueeze(0))[0]
            _ = self.event_process.step()
            fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon).flatten()
            state = torch.cat((obs, torch.tensor(fixed_event_history)), dim=0).unsqueeze(0).float()
            reward = self.reward_fn(state[0], action, self.env)

            if abs(reward)<1e-6:
                rewards[action_idx] = reward
                continue

            for i in range(self.depth - 1):
                action = self.actor(state)
                obs = self.transition_model(state, action)[0]
                _ = self.event_process.step()
                fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon).flatten()
                state = torch.cat((obs, torch.tensor(fixed_event_history)), dim=0).unsqueeze(0).float()
                reward += self.reward_fn(state[0], action[0], self.env)

                if abs(reward)<1e-6:
                    break

            rewards[action_idx] = reward

        best_action_idx = torch.argmax(rewards)

        return actions[best_action_idx]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



# Intelligent look-ahead policy that takes transition and event distributions separately
class StationaryIntelligentLookaheadPolicy(nn.Module):
    def __init__(self, env, obs_dim, event_dim, event_horizon, action_dim, actor, event_process, transition_model, reward_fn, perturbation=0.2, root_degree=5, depth=5, num_trajectories=10, device=torch.device('cpu')):
        super(StationaryIntelligentLookaheadPolicy, self).__init__()
        # self.actor = net.FeedforwardNeuralNetModel(state_dim, 400, action_dim)
        self.actor = actor
        self.event_process = event_process
        self.env = env
        self.obs_dim = obs_dim
        self.event_dim = event_dim
        self.event_horizon = event_horizon
        self.action_dim = action_dim
        self.transition_model = transition_model
        self.reward_fn = reward_fn
        self.perturbation = perturbation
        self.root_degree = root_degree
        self.depth = depth
        self.num_trajectories = num_trajectories
        self.device = device
        self.trajectory_warmup = 5

    def forward(self, start_obs, event_history):
        # Instead of just returning the actor's output
        # we simulate the transition model and reward model for a few time steps
        # and choose action that gives best expected return

        # Initialize the event process with the event history till now
        initial_history = event_history.copy()
        self.event_process.init_history(event_history)

        # Simulate the transition model, event process and reward model for a few time steps
        fixed_event_history = utils.get_fixed_length_vector(event_history, self.event_horizon).flatten()
        start_state = torch.cat((start_obs, torch.tensor(fixed_event_history).to(self.device)), dim=0).unsqueeze(0).float()
        action_perturbation = torch.randn(self.root_degree+1, self.action_dim)
        action_perturbation[0]*=0
        actions = self.actor(start_state[:, :self.obs_dim]).squeeze(0).repeat(self.root_degree+1, 1) + action_perturbation.to(device=self.device) * self.perturbation
        actions = torch.where(actions>1, 2-actions, actions)
        actions = torch.where(actions<-1, -2-actions, actions)
        rewards =  torch.zeros(self.root_degree+1)

        for action_idx in range(self.root_degree+1):
            for _ in range(self.num_trajectories):

                action = actions[action_idx]
                action = torch.clip(action, -1.0, 1.0)
                reward = self.reward_fn(start_state[0].cpu().numpy(), action, self.env)

                if abs(reward)<1e-6:
                    rewards[action_idx] += reward
                    continue

                obs = self.transition_model(start_state, action.unsqueeze(0))[0]
                self.event_process.init_history(initial_history)
                _ = self.event_process.step()
                fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon).flatten()
                state = torch.cat((obs, torch.tensor(fixed_event_history, device=self.device)), dim=0).unsqueeze(0).float()
                reward += self.reward_fn(state[0].cpu().numpy(), action, self.env)

                for i in range(self.depth - 1):
                    if i<self.trajectory_warmup:
                        action = actions[action_idx].unsqueeze(0)
                    else:
                        action = self.actor(state[:, :self.obs_dim])

                    action = self.actor(state[:, :self.obs_dim])
                    action = torch.clip(action, -1.0, 1.0)
                    obs = self.transition_model(state, action)[0]
                    _ = self.event_process.step()
                    fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon).flatten()
                    state = torch.cat((obs, torch.tensor(fixed_event_history, device=self.device)), dim=0).unsqueeze(0).float()
                    reward += self.reward_fn(state[0].cpu().numpy(), action[0], self.env)

                    if abs(reward)<1e-6:
                        break

                rewards[action_idx] += reward

        best_action_idx = torch.argmax(rewards)

        if best_action_idx!=0:
            if (rewards[best_action_idx]-rewards[0])<0.01:
                best_action_idx=0
            else:
                pass

        # best_action_idx = 0
        return actions[best_action_idx]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
