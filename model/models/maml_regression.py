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

def inner_train_step(model, support_data, label, args):
    """ Inner training step procedure. """
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    
    for ii in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.mse_loss(ypred, label)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)

    return updated_params

class MAML(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.model_type == 'ConvNet':
            from model.networks.convnet_maml import ConvNet
            self.encoder = ConvNet(args.way)
        elif args.model_type == 'ResNet':
            from model.networks.resnet_maml import ResNet
            self.encoder = ResNet(args.way)
        elif args.model_type == 'MLP':
            from model.networks.mlp import MLP
            self.encoder = MLP()
        else:
            raise ValueError('')

        self.args = args

    def forward(self, data_shot, data_query, label_shot, mode = 'test'):
        # update with gradient descent
        updated_params = inner_train_step(self.encoder, data_shot, label_shot, self.args)
        # print(update_params)
        if self.args.multi_stage and mode == 'train':
            logitis = []
            for e in updated_params:
                logitis.append(self.encoder(data_query, e) / self.args.temperature)
        else:
            logitis = [self.encoder(data_query, updated_params) / self.args.temperature]
        return logitis