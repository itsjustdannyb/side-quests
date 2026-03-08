import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, latnet_dim:int=100):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(in_features=latnet_dim, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=512)
        self.fc4 = nn.Linear(in_features=512, out_features=784)


    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
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

class CondGenerator(nn.Module):
    def __init__(self, latnet_dim:int=100, num_classes:int=10):
        super(CondGenerator, self).__init__()

        # add embedding layer
        self.embed_layer = nn.Embedding(num_classes, num_classes)

        self.fc1 = nn.Linear(in_features=latnet_dim+num_classes, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=256)
        self.fc5 = nn.Linear(in_features=256, out_features=512)
        self.fc6 = nn.Linear(in_features=512, out_features=784)



    def forward(self, x, labels):
        batch_size = x.shape[0]
        embeddings = self.embed_layer(labels)
        x = F.relu(self.fc1(torch.cat([x, embeddings], dim=1)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.tanh(self.fc6(x))
        return x.reshape(batch_size, 1, 28, 28)


class CondDiscriminator(nn.Module):
    def __init__(self, num_classes:int=10):
        super(CondDiscriminator, self).__init__()

        self.embed_layer = nn.Embedding(num_classes, num_classes)
        self.conv1 = nn.Conv2d(in_channels=1+num_classes, out_channels=8, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3)
        self.fc1 = nn.Linear(9216, 128)
        self.out = nn.Linear(128, 1)
        self.flat = nn.Flatten()


    def forward(self, x, labels):
        batch_size = x.shape[0]
        # shape [batch, num_classes] -> [batch, num_classes, 1, 1] -> [batch, num_classes, 28, 28]
        embeds = self.embed_layer(labels).view(batch_size, -1, 1, 1)
        broad_cast_embeds = embeds.expand(-1, -1, 28, 28)
        
        x = F.relu(self.conv1(torch.cat([x, broad_cast_embeds], dim=1)))
        x = F.relu(self.conv2(x))
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x