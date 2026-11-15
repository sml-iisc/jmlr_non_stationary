import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
import os

from src.models.networks import PolicyNetWithHistory, ValueNetWithHistory

# this one is working good. Tested and it almost converged
# PPO agent for CarRacing-v3 environment

class PolicyGradientAgent:
    def __init__(self, input_dim, nf, output_dim, history_length=10, gamma=0.95, clip_eps=0.2, entropy_coef=0.01, lr=5e-4, device='cuda:0', model_dir='models'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.policy_net = PolicyNetWithHistory(input_dim, nf, output_dim, history_length).to(self.device)
        self.value_net = ValueNetWithHistory(input_dim, nf, history_length).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

        self.model_dir = model_dir

        os.makedirs(model_dir, exist_ok=True)

        self.history_length = history_length

    def select_action(self, obs, hist, argmax=False):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device) / 255.0
        hist = torch.tensor(hist, dtype=torch.float32).to(self.device)  # Add batch dimension

        with torch.no_grad():
            probs = self.policy_net(obs, hist)

        actions = torch.argmax(probs, dim=-1).cpu().numpy() if argmax else torch.multinomial(probs, num_samples=1).squeeze(-1).cpu().numpy()

        return actions, probs[torch.arange(len(actions)), actions].cpu().numpy()

    def get_trajectory(self, envs, outputs=None, argmax=False, termination_reward=-100):
        obs, info = envs.reset()
        obs = np.transpose(obs, (0, 1, 4, 2, 3))
        obs = np.reshape(obs, (obs.shape[0], -1, obs.shape[3], obs.shape[4]))
        traj = [[] for _ in range(envs.num_envs)]
        total_reward, steps = [0 for _ in range(envs.num_envs)], 1000

        # global_terminated, global_truncated = [False for _ in range(envs.num_envs)], [False for _ in range(envs.num_envs)]
        done = [False for _ in range(envs.num_envs)]

        while True:
            dummy_events = [0 for _ in range(1000-steps)]
            hist = info['events']
            hist = [events if (len(events)+steps==1000) else dummy_events for events in hist]

            try:
                hist = np.stack(hist)
            except:
                for i in range(envs.num_envs):
                    print(len(info['events'][i]))

                print(done)
                exit()

            if hist.shape[1] < self.history_length:
                pad_amt = self.history_length - hist.shape[1]
                hist = np.pad(hist, ((0,0), (pad_amt, 0)))
            else:
                hist = hist[:,-self.history_length:]

            actions, prob = self.select_action(obs, hist, argmax)
            next_obs, reward, terminated, truncated, info = envs.step(actions)
            next_obs = np.transpose(next_obs, (0, 1, 4, 2, 3))
            next_obs = np.reshape(next_obs, (next_obs.shape[0], -1, next_obs.shape[3], next_obs.shape[4]))

            for i in range(envs.num_envs):
                if done[i]:
                    continue

                # global_terminated[i] = terminated[i]
                # global_truncated[i] = truncated[i]
                done[i] = terminated[i] or truncated[i] or (total_reward[i] < termination_reward)

                if done[i] and not (terminated[i] or truncated[i]):
                    reward[i] -= 0.1 * steps

                traj[i].append((obs[i].copy(), actions[i], reward[i], done[i], float(prob[i]), hist[i].copy()))
                total_reward[i] += reward[i]

            obs = next_obs

            steps -= 1

            if all(done):
                break


        flat_traj = [sample for subtraj in traj for sample in subtraj]

        return flat_traj

    def collect_trajectories(self, envs, max_steps=8192):
        trajectories = []

        while len(trajectories) < max_steps:
            trajs = self.get_trajectory(envs, argmax=False)
            trajectories.extend(trajs)

        return trajectories[:max_steps]

    def update(self, trajectories, batch_size=128, epochs=5):
        states, actions, rewards, dones, old_probs, hists = zip(*trajectories)

        discounted = []
        R = 0

        for r, d in zip(reversed(rewards), reversed(dones)):
            R = r + self.gamma * R * (1 - d)
            discounted.insert(0, R)

        states = torch.tensor(np.array(states), dtype=torch.float32) / 255.0
        actions = torch.tensor(actions, dtype=torch.int64)
        discounted = torch.tensor(discounted, dtype=torch.float32)
        old_probs = torch.tensor(old_probs, dtype=torch.float32)
        hists = torch.tensor(np.array(hists), dtype=torch.float32).to(self.device)

        discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

        self.policy_net.train()

        for _ in range(epochs):
            for i in range(0, len(states), batch_size):
                s = states[i:i+batch_size].to(self.device)
                a = actions[i:i+batch_size].to(self.device)
                R = discounted[i:i+batch_size].to(self.device)
                old_p = old_probs[i:i+batch_size].to(self.device)
                h = hists[i:i+batch_size].to(self.device)

                with torch.no_grad():
                    v = self.value_net(s, h).squeeze()

                adv = R - v.detach()
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                probs = self.policy_net(s, h)
                dist = probs.gather(1, a.unsqueeze(1)).squeeze()
                ratio = dist / old_p

                clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                loss = -torch.min(ratio * adv, clipped).mean()

                entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1).mean()
                loss -= self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
                self.optimizer.step()

                self.value_net.train()
                v_pred = self.value_net(s, h).squeeze()
                v_loss = F.mse_loss(v_pred, R)
                self.value_optimizer.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
                self.value_optimizer.step()

    def evaluate(self, env, episodes=5, argmax=True):
        self.policy_net.eval()
        rewards = []

        for _ in range(episodes):
            traj = self.get_trajectory(env, argmax=argmax)
            dones = [sample[3] for sample in traj]
            reward = 0

            for j in range(len(dones)):
                reward += traj[j][2]

                if dones[j]:
                    rewards.append(reward)
                    reward = 0

        return np.mean(rewards), np.std(rewards)

    def train(self, envs, eval_envs, start_episode=0, num_episodes=1000, rollout_steps=8192, batch_size=128, epochs=5, eval_interval=10):
        for ep in range(start_episode, num_episodes):
            # print(f"Starting episode - {ep+1}")
            # start_time = time.time()
            trajs = self.collect_trajectories(envs, max_steps=rollout_steps)
            # print(f"Collected {len(trajs)} samples in {time.time() - start_time:.2f} seconds")
            self.update(trajs, batch_size=batch_size, epochs=epochs)

            if (ep+1) % eval_interval == 0:
                print(f"[Episode {ep+1}] Eval...")
                mean, std = self.evaluate(eval_envs, episodes=1, argmax=True)
                print(f"Greedy Eval: {mean:.2f} ± {std:.2f}")
                mean, std = self.evaluate(eval_envs, episodes=1, argmax=False)
                print(f"Stochastic Eval: {mean:.2f} ± {std:.2f}")

                print("Saving model...")
                torch.save(self.policy_net.state_dict(), os.path.join(self.model_dir, f"policy_net_{ep+1}.pth"))
                torch.save(self.value_net.state_dict(), os.path.join(self.model_dir, f"value_net_{ep+1}.pth"))
                print("Model saved.")

    def load(self, policy_path, value_path):
        self.policy_net.load_state_dict(torch.load(policy_path, map_location=self.device, weights_only=True))
        self.value_net.load_state_dict(torch.load(value_path, map_location=self.device, weights_only=True))
