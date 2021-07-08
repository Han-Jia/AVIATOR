import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict
from model.networks.mlp_maml import MLP

# generate top layer classifier based on FC

def update_params(loss, params, acc_gradients, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad
        # accumulate gradients for final updates
        if name == 'layer4.weight':
            acc_gradients[0] += grad
        if name == 'layer4.bias':
            acc_gradients[1] += grad

    return updated_params, acc_gradients

def inner_train_step(model, support_data, label, args):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    acc_gradients = [torch.zeros_like(updated_params['layer4.weight']), torch.zeros_like(updated_params['layer4.bias'])]
    
    for ii in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.mse_loss(ypred, label)
        updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=args.gd_lr, first_order=True)
    return updated_params, acc_gradients

    

class MAML_MSE(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.encoder = MLP()
        self.args = args
        

    def forward(self, data_shot, data_query, label_shot):
        # obtain classifier initializer based on current support data embedding
        support = self.encoder(data_shot, embedding=True)
        support = torch.cat([support, torch.ones(support.size(0), 1)], 1)
        classifier_init = (support.transpose(0, 1).mm(support)+qtorch.eye(support.size(1))).inverse()\
        .mm(support.transpose(0, 1)).mm(label_shot).view(1, -1)
        
        # split init into weight and bias
        fc_weight_init, fc_bias_init = classifier_init.split(support.size(1)-1, 1)
        # set the initial classifier
        self.encoder.layer4.weight.data = fc_weight_init.data
        self.encoder.layer4.bias.data = fc_bias_init.squeeze().data
        
        # update with gradient descent
        updated_params, acc_gradients = inner_train_step(self.encoder, data_shot, label_shot, self.args)
        
        # reupate with the initial classifier and the accumulated gradients
        updated_params['layer4.weight'] = fc_weight_init.squeeze(0) - self.args.gd_lr * acc_gradients[0]
        updated_params['layer4.bias'] = fc_bias_init.squeeze() - self.args.gd_lr * acc_gradients[1]
        
        logitis = [self.encoder(data_query, updated_params)]
        return logitis



