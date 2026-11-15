import torch.nn as nn
import torch

class PolicyNet(nn.Module):
    def __init__(self, input_dim, nf, output_dim):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nf, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=1, num_channels=nf)
        self.fc3 = nn.Linear(nf*6*6, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = x.contiguous()
        x = torch.nn.functional.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        x = torch.nn.functional.elu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.elu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.elu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return self.softmax(x)


class ValueNet(nn.Module):
    def __init__(self, input_dim, nf):
        super(ValueNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nf, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv4 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.fc3 = nn.Linear(nf*6*6, 1)

    def forward(self, x):
        x = x.contiguous()
        x = torch.nn.functional.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        x = torch.nn.functional.elu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.elu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.elu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc3(x)

        return x


class PolicyNetWithHistory(nn.Module):
    def __init__(self, input_dim, nf, output_dim, history_length=4):
        super(PolicyNetWithHistory, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nf, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=1, num_channels=nf)

        intermediate_dim = max(history_length, nf)
        self.fc_h1 = nn.Linear(history_length, intermediate_dim)  # Fully connected layer to process history
        self.fc_h2 = nn.Linear(intermediate_dim, nf*6*6)  # Another fully connected layer for further processing
        self.fc3 = nn.Linear(nf*6*6*2, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, history):
        x = x.contiguous()
        x = torch.nn.functional.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        x = torch.nn.functional.elu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.elu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.elu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        h_x = self.fc_h1(history)  # Process history
        h_x = torch.nn.functional.elu(h_x)
        h_x = self.fc_h2(h_x)
        x_comb = torch.cat((x, h_x), dim=1)  # Concatenate processed history with CNN output
        x = self.fc3(x_comb)

        return self.softmax(x)

class ValueNetWithHistory(nn.Module):
    def __init__(self, input_dim, nf, history_length=4):
        super(ValueNetWithHistory, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, nf, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.GroupNorm(num_groups=1, num_channels=nf)  # Using GroupNorm instead of BatchNorm
        self.conv3 = nn.Conv2d(nf, nf, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.GroupNorm(num_groups=1, num_channels=nf)

        intermediate_dim = max(history_length, nf)
        self.fc_h1 = nn.Linear(history_length, intermediate_dim)  # Fully connected layer to process history
        self.fc_h2 = nn.Linear(intermediate_dim, nf*6*6)  # Another fully connected layer for further processing
        self.fc3 = nn.Linear(nf*6*6*2, 1)

    def forward(self, x, history):
        x = x.contiguous()
        x = torch.nn.functional.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        x = torch.nn.functional.elu(self.bn1(self.conv1(x)))
        x = torch.nn.functional.elu(self.bn2(self.conv2(x)))
        x = torch.nn.functional.elu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        h_x = self.fc_h1(history)  # Process history
        h_x = torch.nn.functional.elu(h_x)
        h_x = self.fc_h2(h_x)
        x_comb = torch.cat((x, h_x), dim=1)  # Concatenate processed history with CNN output
        x = self.fc3(x_comb)

        return x
