import torch
import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        # First convolutional layer 
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional layer 
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Third convolutional layer 
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        
        # Fourth convolutional layer 
        self.conv4 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fully connected layer
        self.fc1 = nn.Linear(16 * 3 * 3, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))        # 28x28 -> 28x28
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 28x28 -> 14x14
        x = self.relu(self.bn3(self.conv3(x)))        # 14x14 -> 14x14
        x = self.pool(self.relu(self.bn4(self.conv4(x))))  # 14x14 -> 7x7
        x = self.pool(x)                              # 7x7 -> 3x3
        x = x.view(-1, 16 * 3 * 3)
        x = self.fc1(x)
        return x