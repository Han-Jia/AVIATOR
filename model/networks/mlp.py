import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class MLP(nn.Module):

    def __init__(self, num_hidden=100):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(2, num_hidden)
        self.layer2 = nn.Linear(num_hidden, num_hidden)
        self.layer3 = nn.Linear(num_hidden, num_hidden)
        self.layer4 = nn.Linear(num_hidden, 1)

    def forward(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
                    
        x = F.linear(x, weight=params['layer1.weight'], bias=params['layer1.bias'])
        x = F.relu(x)
        x = F.linear(x, weight=params['layer2.weight'], bias=params['layer2.bias'])
        x = F.relu(x)
        x = F.linear(x, weight=params['layer3.weight'], bias=params['layer3.bias'])
        x = F.relu(x)
        

        if not embedding:
            x = F.linear(x, weight=params['layer4.weight'], bias=params['layer4.bias'])    

        return x
    
