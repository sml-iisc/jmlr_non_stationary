from pathlib import Path
import os
import random
import numpy as np
import torch
from src.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
RUN_DIR = Path(os.environ.get("RUN_DIR", ROOT / "runs")).resolve()
os.chdir(ROOT)
torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '1')))
random.seed(int(os.environ.get('SEED', '0')))
np.random.seed(int(os.environ.get('SEED', '0')))
torch.manual_seed(int(os.environ.get('SEED', '0')))

import torch
from src.environment import legacy_api as gym
import numpy as np

from src.models.networks import FeedforwardNeuralNetModel, FeedforwardHybridNeuralNetModel
from src.training.replay import Buffer

import matplotlib.pyplot as plt

############# Hyperparameters #############

hidden_dim = 400
gamma = 0.95
num_episodes = int(os.environ.get('TRAIN_EPISODES', '10000'))
batch_size = 256
buffer_size = 100000
eval_episodes = int(os.environ.get('EVAL_EPISODES', '10'))
eval_interval = int(os.environ.get('EVAL_INTERVAL', '10'))
learning_rate_critic = 0.001
learning_rate_actor = 0.0001
tau = 0.01
exploration_variance = 0.3
model_save_path = str(RUN_DIR / 'actor')
os.makedirs(model_save_path, exist_ok=True)
model_save_suffix = ''
model_save_interval = 10

loss_print_interval = 1000

device = torch.device('cpu')


################ Algorithm ################

# env = gym.make('HalfCheetah-v2')
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.sample().shape[0]
action_dim = env.action_space.sample().shape[0]

Q = FeedforwardHybridNeuralNetModel(state_dim, action_dim, hidden_dim, 1)
policy = FeedforwardNeuralNetModel(state_dim, hidden_dim, action_dim)

Q_target = FeedforwardHybridNeuralNetModel(state_dim, action_dim, hidden_dim, 1)
policy_target = FeedforwardNeuralNetModel(state_dim, hidden_dim, action_dim)

replay_buffer = Buffer(state_dim=state_dim, action_dim=action_dim, capacity=buffer_size, batch_size=batch_size, device=device)

if 'cuda' in device.type:
    Q.cuda()
    Q_target.cuda()
    policy.cuda()
    policy_target.cuda()

loss_critic = torch.nn.MSELoss()
optimizer_critic = torch.optim.Adam(Q.parameters(), lr=learning_rate_critic)
optimizer_actor = torch.optim.Adam(policy.parameters(), lr=learning_rate_actor)

n_updates = 0
eval_avg_episode_rewards = []

for episode in range(num_episodes):

    # Initialize the environment
    state = torch.tensor(env.reset()).to(device=device).float()
    done = False
    episode_reward = 0.0

    while not done:
        action = policy(state).detach()
        action += (torch.randn_like(action) * exploration_variance)
        next_state, reward, done, info = env.step(action.cpu().numpy())
        episode_reward += reward
        reward = torch.asarray([reward], device=device).float()
        next_state = torch.asarray(next_state, device=device).float()
        done = torch.asarray([done], device=device).float()
        replay_buffer.add_transition(state, action, reward, next_state, done)
        state = next_state

        if len(replay_buffer) > batch_size:
            # Update all the networks

            (state_batch, action_batch, reward_batch, next_state_batch, done_batch) = replay_buffer.sample(num_samples=batch_size)

            # Update the critic
            policy_action_batch = policy_target(next_state_batch)
            q_target = reward_batch.view(-1, 1) + gamma * Q_target(next_state_batch, policy_action_batch)
            # print(q_target.shape)
            # print('r', reward_batch.shape)
            # print(';', next_state_batch.shape)
            # print(':', policy_action_batch.shape)
            loss = loss_critic(q_target, Q(state_batch, action_batch))


            optimizer_critic.zero_grad()
            loss.backward()
            optimizer_critic.step()

            # Update the actor

            policy_action_batch = policy(state_batch)
            q_value = Q(state_batch, policy_action_batch)
            actor_objective = -torch.mean(q_value)

            optimizer_actor.zero_grad()
            actor_objective.backward()
            optimizer_actor.step()


            # Update target networks
            for q, q_prime in zip(Q.parameters(), Q_target.parameters()):
                q_prime.data = tau * q.data + (1 - tau) * q_prime.data

            for p, p_prime in zip(policy.parameters(), policy_target.parameters()):
                p_prime.data = tau * p.data + (1 - tau) * p_prime.data

            n_updates += 1

            if n_updates % loss_print_interval == 0:
                print('Critic loss:', loss)

    if episode % eval_interval == 0:
        eval_avg_episode_reward = 0.0

        for _ in range(eval_episodes):
            eval_episode_reward = 0.0
            state = torch.tensor(env.reset()).to(device=device).float()
            done = False

            while not done:
                action = policy(state).detach().cpu().numpy()
                next_state, reward, done, _ = env.step(action)
                eval_episode_reward += reward
                state = torch.asarray(next_state, device=device).float()

            eval_avg_episode_reward += eval_episode_reward

        eval_avg_episode_reward /= eval_episodes
        print('Updates: {}; Training episode {}: Reward {:.2f}; Evaluation average episode reward: {:.2f}'\
            .format(n_updates, episode, episode_reward, eval_avg_episode_reward))
        eval_avg_episode_rewards.append(eval_avg_episode_reward)

    if episode % model_save_interval == 0:
        torch.save(policy, model_save_path + '/policy' + model_save_suffix + str(episode))
        torch.save(policy_target, model_save_path + '/policy_target' + model_save_suffix + str(episode))
        torch.save(Q, model_save_path + '/Q' + model_save_suffix + str(episode))
        torch.save(Q_target, model_save_path + '/Q_target' + model_save_suffix + str(episode))

# Plot the average reward per episode
plt.plot(eval_avg_episode_rewards)
plt.savefig(RUN_DIR / "actor/evaluation_rewards.png")

# Portable final checkpoint for ACTOR_CHECKPOINT.
torch.save(policy, os.path.join(model_save_path, "policy_final"))


env.close()
plt.close("all")
