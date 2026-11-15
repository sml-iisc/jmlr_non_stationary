import torch
import torch.nn as nn

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(out)
        out = self.relu2(out)

        out = self.fc3(out)
        out = 2 * self.tanh(out)

        return out

class FeedforwardHybridNeuralNetModel(nn.Module):
    def __init__(self, input_1_dim, input_2_dim, hidden_dim, output_dim):
        super(FeedforwardHybridNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_1_dim, hidden_dim) 
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim + input_2_dim, hidden_dim)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x, y):
        out = self.fc1(x)
        out = self.relu1(out)

        out = self.fc2(torch.cat((out, y), dim=1))
        # print('out', out.shape)
        out = self.relu2(out)

        out = self.fc3(out)

        return out
