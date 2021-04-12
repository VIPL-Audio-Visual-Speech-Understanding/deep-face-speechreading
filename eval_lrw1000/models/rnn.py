import math
import torch
import torch.nn as nn
from torch.nn import init


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, every_frame=True):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.every_frame = every_frame
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

        # init
        stdv = math.sqrt(2 / (input_size + hidden_size))
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                for i in range(0, hidden_size * 3, hidden_size):
                    nn.init.uniform_(param.data[i: i + hidden_size],
                                  -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            elif 'weight_hh' in name:
                for i in range(0, hidden_size * 3, hidden_size):
                    nn.init.orthogonal_(param.data[i: i + hidden_size])
            elif 'bias' in name:
                for i in range(0, hidden_size * 3, hidden_size):
                    nn.init.constant_(param.data[i: i + hidden_size], 0)

    def forward(self, x):
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        if self.every_frame:
            out = self.fc(out)  # predictions based on every time step
        else:
            out = self.fc(out[:, -1, :])  # predictions based on last time-step

        return out
