import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latnet_dim:int=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=latnet_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=784)


    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x.reshape(batch_size, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self ):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.out = nn.Linear(128, 1)
        self.flat = nn.Flatten()


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x