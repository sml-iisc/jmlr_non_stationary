import torch
from torch.utils.data import Dataset

class Buffer(Dataset):
    def __init__(self, state_dim, action_dim, capacity=10000, batch_size=256, device=torch.device('cpu')) -> None:
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device

        self.state = torch.zeros((0, self.state_dim), device=self.device)
        self.action = torch.zeros((0, self.action_dim), device=self.device)
        self.next_state = torch.zeros((0, self.state_dim), device=self.device)
        self.reward = torch.zeros(0, device=self.device)
        self.done = torch.zeros(0, device=self.device)

        self.size = 0
        self.index = 0

    def add_transition(self, state, action, reward, next_state, done):

        if self.size < self.capacity:
            self.state = torch.cat((self.state, torch.asarray(state).view(1, -1).to(device=self.device)), dim=0)
            self.action = torch.cat((self.action, torch.asarray(action).view(1, -1).to(device=self.device)), dim=0)
            self.reward = torch.cat((self.reward, torch.asarray(reward)))
            self.next_state = torch.cat((self.next_state, torch.asarray(next_state).view(1, -1).to(device=self.device)), dim=0)
            self.done = torch.cat((self.done, torch.asarray(done)))
            self.size += 1

        else:
            self.state[self.index] = torch.asarray(state).view(-1)
            self.action[self.index] = torch.asarray(action).view(-1)
            self.next_state[self.index] = torch.asarray(next_state).view(-1)
            self.reward[self.index] = torch.asarray(reward).view(-1)
            self.done[self.index] = torch.asarray(done).view(-1)

        self.index = (self.index + 1) % self.capacity

    def sample(self, num_samples=0):
        num_samples = num_samples if num_samples > 0 else self.batch_size
        idx = torch.randint(low=0, high=self.size, size=(num_samples,), device=self.device)

        return (self.state[idx], self.action[idx], self.reward[idx], self.next_state[idx], self.done[idx])

    def __getitem__(self, index):
        return (self.state[index], self.action[index], self.reward[index], self.next_state[index], self.done[index])

    def __len__(self):
        return self.size
