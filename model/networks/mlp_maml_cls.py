import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class MLP(nn.Module):

    def __init__(self, num_hidden=16):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 2)
        self.FC = nn.Linear(2, 3)

    def forward(self, x, params=None, embedding=False):
        if params is None:
            params = OrderedDict(self.named_parameters())
                    
        x = F.linear(x, weight=params['layer1.weight'], bias=params['layer1.bias'])
        x = F.relu(x)
        x = F.linear(x, weight=params['layer2.weight'], bias=params['layer2.bias'])
        x = F.relu(x)

        if not embedding:
            x = F.linear(x, weight=params['FC.weight'], bias=params['FC.bias'])

        return x
    