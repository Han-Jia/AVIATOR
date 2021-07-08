import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict

# generate top layer classifier based on FC

def update_params(loss, params, acc_gradients, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad
        # accumulate gradients for final updates
        if name == 'FC.weight':
            acc_gradients[0] = acc_gradients[0] + grad
        if name == 'FC.bias':
            acc_gradients[1] = acc_gradients[1] + grad

    return updated_params, acc_gradients

def inner_train_step(model, support_data, label, args):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    acc_gradients = [torch.zeros_like(updated_params['FC.weight']), torch.zeros_like(updated_params['FC.bias'])]
    
    for ii in range(args.inner_iters):
        ypred = model(support_data, updated_params)
        loss = F.mse_loss(ypred, label)
        updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=args.gd_lr, first_order=True)
    return updated_params, acc_gradients


class MAML_lse_regression(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.model_type == 'ConvNet':
            from model.networks.convnet_maml import ConvNet
            self.encoder = ConvNet(args.way)
            self.h_dim = h_dim = 64
        elif args.model_type == 'ResNet':
            from model.networks.resnet_maml import ResNet
            self.encoder = ResNet(args.way)
            self.h_dim = h_dim = 640
        else:
            raise ValueError('')

        self.args = args
        # construct FC to generate task-dependent classifier
        self.fc1 = nn.Linear(h_dim * 2, min(h_dim * 4, 1024))
        self.fc2 = nn.Linear(min(h_dim * 4, 1024), h_dim + 1) # 1 for bias
        

    def forward(self, data_shot, data_query):
        # obtain classifier initializer based on current support data embedding
        support = self.encoder(data_shot, embedding=True)
        
        classifier_init = self.fc2(F.relu(self.fc1(combined_input)))
        
        # split init into weight and bias
        fc_weight_init, fc_bias_init = classifier_init.split(self.h_dim, 1)
        # set the initial classifier
        self.encoder.FC.weight.data = fc_weight_init.data
        self.encoder.FC.bias.data = fc_bias_init.squeeze().data
        
        # update with gradient descent
        updated_params, acc_gradients = inner_train_step(self.encoder, data_shot, self.args)
        
        # reupate with the initial classifier and the accumulated gradients
        updated_params['FC.weight'] = fc_weight_init.squeeze(0) - self.args.gd_lr * acc_gradients[0]
        updated_params['FC.bias'] = fc_bias_init.squeeze() - self.args.gd_lr * acc_gradients[1]
        
        logitis = self.encoder(data_query, updated_params) / self.args.temperature
        return logitis



