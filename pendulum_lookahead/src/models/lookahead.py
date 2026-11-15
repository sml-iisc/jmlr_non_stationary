# Create a policy that encapsulates an actor network

import torch
import torch.nn as nn
import torch.optim as optim

from src.models import networks as net
from src.utils import features as utils

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
    def __init__(self, env, obs_dim, event_dim, event_horizon, action_dim, actor, transition_model, reward_fn, perturbation=0.2, root_degree=5, depth=5):
        super(StationaryNaiveLookaheadPolicy, self).__init__()
        # self.actor = net.FeedforwardNeuralNetModel(state_dim, 400, action_dim)
        self.actor = actor
        self.env = env
        self.obs_dim = obs_dim
        self.event_dim = event_dim
        self.event_horizon = event_horizon
        self.state_dim = obs_dim + event_dim * event_horizon
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
        obs = state[:self.obs_dim].unsqueeze(0)
        state = state.unsqueeze(0)
        actions = self.actor(obs).squeeze(0).repeat(self.root_degree, 1) + torch.randn(self.root_degree, self.action_dim) * self.perturbation

        rewards = torch.zeros(self.root_degree)

        for action_idx in range(self.root_degree):
            action = actions[action_idx]
            state = self.transition_model(state, action.unsqueeze(0))
            reward = self.reward_fn(state[0].cpu().detach().numpy(), action.detach().cpu().numpy(), self.env)

            for i in range(self.depth - 1):
                action = self.actor(state[:, :self.obs_dim])
                state = self.transition_model(state, action)
                reward += self.reward_fn(state.cpu().detach().numpy()[0], action.cpu().detach().numpy()[0], self.env)

            rewards[action_idx] = reward

        best_action_idx = torch.argmax(rewards)

        return actions[best_action_idx]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



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
        fixed_event_history = utils.get_fixed_length_vector(event_history, self.event_horizon)
        state = torch.cat((obs, torch.tensor(fixed_event_history)), dim=0).unsqueeze(0).float()
        actions = self.actor(state).squeeze(0).repeat(self.root_degree, 1) + torch.randn(self.root_degree, self.action_dim) * self.perturbation

        rewards = torch.zeros(self.root_degree)

        for action_idx in range(self.root_degree):
            action = actions[action_idx]
            obs = self.transition_model(state, action.unsqueeze(0))[0]
            _ = self.event_process.step()
            fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon)
            state = torch.cat((obs, torch.tensor(fixed_event_history)), dim=0).unsqueeze(0).float()
            reward = self.reward_fn(state[0].cpu().detach().numpy(), action.detach().cpu().numpy(), self.env)

            for i in range(self.depth - 1):
                action = self.actor(state)
                obs = self.transition_model(state, action)[0]
                _ = self.event_process.step()
                fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon)
                state = torch.cat((obs, torch.tensor(fixed_event_history)), dim=0).unsqueeze(0).float()
                reward += self.reward_fn(state.cpu().detach().numpy()[0], action.cpu().detach().numpy()[0], self.env)

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

    def forward(self, obs, event_history):
        # Instead of just returning the actor's output
        # we simulate the transition model and reward model for a few time steps
        # and choose action that gives best expected return

        # Initialize the event process with the event history till now
        initial_history = event_history.copy()
        self.event_process.init_history(event_history)

        # Simulate the transition model, event process and reward model for a few time steps
        fixed_event_history = utils.get_fixed_length_vector(event_history, self.event_horizon)
        start_state = torch.cat((obs, torch.tensor(fixed_event_history).to(self.device)), dim=0).unsqueeze(0).float()
        actions = self.actor(start_state[:, :self.obs_dim]).squeeze(0).repeat(self.root_degree, 1) + torch.randn(self.root_degree, self.action_dim).to(device=self.device) * self.perturbation
        rewards =  torch.zeros(self.root_degree)

        for action_idx in range(self.root_degree):
            for _ in range(self.num_trajectories):
                reward = self.reward_fn(start_state[0].cpu().detach().numpy(), actions[action_idx].detach().cpu().numpy(), self.env)
                action = actions[action_idx]
                obs = self.transition_model(start_state, action.unsqueeze(0))[0]
                self.event_process.init_history(initial_history)
                _ = self.event_process.step()
                fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon)
                state = torch.cat((obs, torch.tensor(fixed_event_history, device=self.device)), dim=0).unsqueeze(0).float()
                reward += self.reward_fn(state[0].cpu().detach().numpy(), action.detach().cpu().numpy(), self.env)

                for i in range(self.depth - 1):
                    action = self.actor(state[:, :self.obs_dim])
                    obs = self.transition_model(state, action)[0]
                    _ = self.event_process.step()
                    fixed_event_history = utils.get_fixed_length_vector(self.event_process.event_history, self.event_horizon)
                    state = torch.cat((obs, torch.tensor(fixed_event_history, device=self.device)), dim=0).unsqueeze(0).float()
                    reward += self.reward_fn(state.cpu().detach().numpy()[0], action.cpu().detach().numpy()[0], self.env)

                rewards[action_idx] += reward

        best_action_idx = torch.argmax(rewards)

        return actions[best_action_idx]


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
