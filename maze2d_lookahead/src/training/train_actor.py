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
import gymnasium as gym
import numpy as np
from torch import nn
from torch.optim import Adam
import os
import matplotlib.pyplot as plt

from src.models import networks as net

# Hyperparameters
hidden_dim = 256
gamma = 0.99  # Changed from 1 to 0.99 for discounting
tau = 0.005
actor_lr = 1e-4
critic_lr = 1e-3
batch_size = 256
buffer_size = int(1e6)
num_episodes = int(os.environ.get('TRAIN_EPISODES', '10000'))
expl_noise = 0.1  # Reduced exploration noise
max_action = 1.0  # Assuming action space is [-1, 1]

# Networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Added tanh to bound actions
        )

    def forward(self, state):
        return self.net(state) * max_action  # Scale to action space range

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

# Modified Replay Buffer with GPU support
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size, device):
        self.device = device
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.ptr = 0
        self.size = 0
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = torch.tensor(state, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.tensor(action, dtype=torch.float32, device=self.device)
        self.rewards[self.ptr] = torch.tensor(reward, dtype=torch.float32, device=self.device)
        self.next_states[self.ptr] = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.tensor(done, dtype=torch.float32, device=self.device)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx].unsqueeze(1),
            self.next_states[idx],
            self.dones[idx].unsqueeze(1)
        )


from src.environment.maps import get_map
from gymnasium.envs.registration import register

register(
    id=f"StationaryMaze_v1",
    entry_point="src.environment.maze:StationaryMazeEnv",
    max_episode_steps=300,
)

randomize_goal = True
goal = -1

MAP_ROWS = 10
MAP_COLS = 10

params = {
    'event_dim':1,
    'time_horizon':1000,
    'gravity_centers':10,
    'base_intensity':0.1,
    'alpha_coef':1,
    'beta_coef':1,
    'lamda_intensity':0.02,
    'lamda_means':0.2
}

custom_map, gravity_centers = get_map(MAP_ROWS, MAP_COLS, params['gravity_centers'])

eval_interval = int(os.environ.get('EVAL_INTERVAL', '10'))
eval_episodes = int(os.environ.get('EVAL_EPISODES', '10'))
log_interval = 10          # How often to log progress
save_interval = 1000       # How often to save models

# Create directories
os.makedirs(RUN_DIR / "actor/logs", exist_ok=True)
os.makedirs(str(RUN_DIR / "actor"), exist_ok=True)

# Training Setup
env = gym.make('StationaryMaze_v1', maze_map=custom_map, reward_type='dense', 
               continuing_task=False, randomize_goal=randomize_goal, goal=goal, render_mode=None)

state_dim = env.observation_space['observation'].shape[0] + env.observation_space['desired_goal'].shape[0]
action_dim = env.action_space.shape[0]

# Configure device
device = torch.device('cpu')
print(f"Using device: {device}")

# Modified network initialization with GPU support
actor = net.FeedforwardNeuralNetModel(state_dim, hidden_dim, action_dim).to(device)
critic = net.FeedforwardHybridNeuralNetModel(state_dim, action_dim, hidden_dim, 1).to(device)
target_actor = net.FeedforwardNeuralNetModel(state_dim, hidden_dim, action_dim).to(device)
target_critic = net.FeedforwardHybridNeuralNetModel(state_dim, action_dim, hidden_dim, 1).to(device)
# actor = Actor(state_dim, action_dim).to(device)
# critic = Critic(state_dim, action_dim).to(device)
# target_actor = Actor(state_dim, action_dim).to(device)
# target_critic = Critic(state_dim, action_dim).to(device)

# Initialize target networks
target_actor.load_state_dict(actor.state_dict())
target_critic.load_state_dict(critic.state_dict())


# actor.load_state_dict(torch.load('./runs/actor/actor_9000.pth', weights_only=True))
# target_actor.load_state_dict(torch.load('./runs/actor/actor_9000.pth', weights_only=True))
# critic.load_state_dict(torch.load('./runs/actor/critic_9000.pth', weights_only=True))
# target_critic.load_state_dict(torch.load('./runs/actor/critic_9000.pth', weights_only=True))


optimizer_actor = Adam(actor.parameters(), lr=actor_lr)
optimizer_critic = Adam(critic.parameters(), lr=critic_lr)

# Create buffer with device
buffer = ReplayBuffer(state_dim, action_dim, buffer_size, device)

# Initialize logger
import csv
from datetime import datetime
log_file = f"{RUN_DIR}/actor/logs/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
with open(log_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['episode', 'reward', 'critic_loss', 'actor_loss', 'eval_reward'])

# Training Loop
best_eval_reward = -float('inf')
episode_rewards = []
critic_losses = []
actor_losses = []
eval_rewards = []

# Training Loop
for episode in range(num_episodes):
    obs_dict, _ = env.reset()
    state = np.concatenate([obs_dict['observation'], obs_dict['desired_goal']])
    episode_reward = 0
    done = False
    truncated = False

    total_critic_loss = 0.0
    total_actor_loss = 0.0
    update_count = 0

    while not done and not truncated:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(device)
            action = actor(state_tensor).cpu().numpy()
            # Add noise and clip
            action = np.clip(action + np.random.normal(0, expl_noise, size=action_dim), 
                           -max_action, max_action)

        next_obs_dict, reward, done, truncated, _ = env.step(action)
        next_state = np.concatenate([next_obs_dict['observation'], next_obs_dict['desired_goal']])
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Update networks
        if buffer.size >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample()

            # Critic update
            with torch.no_grad():
                target_actions = target_actor(next_states)
                target_q = target_critic(next_states, target_actions)
                target_q = rewards + gamma * (1 - dones) * target_q

            current_q = critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q)

            optimizer_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)  # Gradient clipping
            optimizer_critic.step()

            # Actor update
            actor_loss = -critic(states, actor(states)).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer_actor.step()

            # Accumulate losses
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            update_count += 1

            # Update target networks
            for param, target_param in zip(critic.parameters(), target_critic.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # Evaluation and logging here
    # Logging
    avg_critic_loss = total_critic_loss / update_count if update_count > 0 else 0
    avg_actor_loss = total_actor_loss / update_count if update_count > 0 else 0
    episode_rewards.append(episode_reward)
    critic_losses.append(avg_critic_loss)
    actor_losses.append(avg_actor_loss)

    # Evaluation
    if episode % eval_interval == 0:
        avg_eval_reward = 0

        for _ in range(eval_episodes):
            eval_obs_dict, _ = env.reset()
            eval_state = np.concatenate([eval_obs_dict['observation'], 
                                      eval_obs_dict['desired_goal']])
            eval_reward = 0
            eval_done = False
            eval_truncated = False

            while not eval_done and not eval_truncated:
                with torch.no_grad():
                    eval_action = target_actor(torch.FloatTensor(eval_state).to(device)).cpu().numpy()

                eval_next_obs_dict, reward, eval_done, eval_truncated, _ = env.step(eval_action)
                eval_state = np.concatenate([eval_next_obs_dict['observation'], 
                                          eval_next_obs_dict['desired_goal']])
                eval_reward += reward

            avg_eval_reward += eval_reward

        avg_eval_reward /= eval_episodes
        eval_rewards.append(avg_eval_reward)

        # Save best model
        if avg_eval_reward > best_eval_reward:
            best_eval_reward = avg_eval_reward
            torch.save(actor.state_dict(), f"{RUN_DIR}/actor/best_actor.pth")
            torch.save(critic.state_dict(), f"{RUN_DIR}/actor/best_critic.pth")

        print(f"Episode {episode}/{num_episodes}")
        print(f"Train Reward: {episode_reward:.2f} | Eval Reward: {avg_eval_reward:.2f}")
        print(f"Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
        print("---------------------------------")

    # Save checkpoints
    if episode % save_interval == 0:
        torch.save(actor.state_dict(), f"{RUN_DIR}/actor/actor_{episode}.pth")
        torch.save(critic.state_dict(), f"{RUN_DIR}/actor/critic_{episode}.pth")

    # Write to CSV
    with open(log_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow([
            episode,
            episode_reward,
            avg_critic_loss,
            avg_actor_loss,
            avg_eval_reward if episode % eval_interval == 0 else None
        ])

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(episode_rewards)
plt.title("Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(eval_rewards)
plt.title("Evaluation Rewards")
plt.xlabel("Evaluation Epoch")
plt.ylabel("Average Reward")

plt.savefig(RUN_DIR / "actor/training_progress.png")
plt.close()

# Save the final state as well as the original periodic checkpoints.
torch.save(actor.state_dict(), RUN_DIR / "actor/actor_final.pth")
torch.save(critic.state_dict(), RUN_DIR / "actor/critic_final.pth")


env.close()
plt.close("all")
