import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict

def update_params(loss, params, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad

    return updated_params

def inner_train_step(model, support_data, args):
    """ Inner training step procedure. """
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)         
    
    for ii in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)
    return updated_params

class MAML(nn.Module):

    def __init__(self, args):
        super().__init__()
        from model.networks.mlp_maml_cls import MLP
        self.encoder = MLP()

        self.args = args

    def forward(self, data_shot, data_query, mode = 'test'):
        # update with gradient descent
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
        if self.args.multi_stage and mode == 'train':
            logitis = []
            for e in updated_params:
                logitis.append(self.encoder(data_query, e) / self.args.temperature)
        else:
            logitis = [self.encoder(data_query, updated_params) / self.args.temperature]
        return logitis