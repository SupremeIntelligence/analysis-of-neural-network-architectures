import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=128*128*1, num_classes=10):
        super().__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, num_classes)

    def forward (self, x):
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        return x

