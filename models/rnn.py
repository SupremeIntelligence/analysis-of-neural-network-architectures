import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size=128, hidden_size=128, num_layers=1, num_classes=6):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):

        x = x.squeeze(1)

        out, hidden = self.rnn(x)

        x = hidden[-1]

        x = self.fc(x)

        return x