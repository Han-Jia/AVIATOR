import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict
from model.utils import euclidean_metric

# generate top layer classifier based on FC

def update_params(loss, params, step_size=0.5, first_order=True):
    name_list, tensor_list = zip(*params.items())
    grads = torch.autograd.grad(loss, tensor_list, create_graph=not first_order)
    updated_params = OrderedDict()
    for name, param, grad in zip(name_list, tensor_list, grads):
        updated_params[name] = param - step_size * grad

    return updated_params

def inner_train_step(model, support_data, args):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)        
    
    # first update
    shot_emb = model(support_data, updated_params, embedding=True)
    # get classifier with ProtoType
    proto = shot_emb.reshape(args.shot, args.way, -1).mean(dim=0)
    ypred = 2 * torch.mm(shot_emb, proto.t()) - torch.sum(proto ** 2, 1).unsqueeze(0)
    loss = F.cross_entropy(ypred, label)
    updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)    
    
    for ii in range(args.inner_iters - 1):
        shot_emb = model(support_data, updated_params, embedding=True)
        # get classifier with ProtoType
        proto = shot_emb.reshape(args.shot, args.way, -1).mean(dim=0)
        ypred = 2 * torch.mm(shot_emb, proto.t()) - torch.sum(proto ** 2, 1).unsqueeze(0)
        loss = F.cross_entropy(ypred, label)
        updated_params = update_params(loss, updated_params, step_size=args.gd_lr, first_order=True)
    return updated_params
    

class ProtoMAML(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.model_type == 'ConvNet':
            from model.networks.convnet_maml import ConvNet
            self.encoder = ConvNet(args.way, last_layer=False)
            self.h_dim = h_dim = 64
        elif args.model_type == 'ResNet':
            from model.networks.resnet_maml import ResNet
            self.encoder = ResNet(args.way)
            self.h_dim = h_dim = 640
        else:
            raise ValueError('')

        self.args = args

    def forward(self, data_shot, data_query):
        # update with gradient descent
        updated_params = inner_train_step(self.encoder, data_shot, self.args)
                
        # for ProtoNet        
        proto = self.encoder(data_shot, updated_params, embedding=True)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        logits = euclidean_metric(self.encoder(data_query, updated_params), proto) / self.args.temperature
            
        return logits