import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from collections import OrderedDict

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

def inner_train_step(model, support_data, args):
    """ Inner training step procedure. 
        Should accumulate and record the gradient"""
    # obtain final prediction
    updated_params = OrderedDict(model.named_parameters())
    acc_gradients = [torch.zeros_like(updated_params['FC.weight']), torch.zeros_like(updated_params['FC.bias'])]
    label = torch.arange(args.way).repeat(args.shot)
    if torch.cuda.is_available():
        label = label.type(torch.cuda.LongTensor)
    else:
        label = label.type(torch.LongTensor)        
    
    # first update
    ypred = model(support_data, updated_params)
    loss = F.cross_entropy(ypred, label)
    updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=args.gd_lr, first_order=True)    
    
    for ii in range(args.inner_iters - 1):
        ypred = model(support_data, updated_params)
        loss = F.cross_entropy(ypred, label)
        updated_params, acc_gradients = update_params(loss, updated_params, acc_gradients, step_size=args.gd_lr, first_order=True)
    return updated_params, acc_gradients

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn


class SimpleAttention(nn.Module):
    ''' One-Layer One-Head Attention module '''

    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k)
        self.w_ks = nn.Linear(d_model, d_k)
        self.w_vs = nn.Linear(d_v, d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_v)

        self.fc = nn.Linear(d_model, 1)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v = self.d_k, self.d_v
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, 1, d_k)
        k = self.w_ks(k).view(sz_b, len_k, 1, d_k)
        v = self.w_vs(v).view(sz_b, len_v, 1, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(1, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.layer_norm(self.dropout(output) + torch.cat([residual, self.fc(residual)], -1))

        return output
    

class MAMLP(nn.Module):

    def __init__(self, args):
        super().__init__()
        if args.model_type == 'ConvNet':
            from model.networks.convnet_maml import ConvNet
            self.encoder = ConvNet(args.way)
            self.h_dim = 64
        elif args.model_type == 'ResNet':
            from model.networks.resnet_maml import ResNet
            self.encoder = ResNet(args.way)
            self.h_dim = 640
        else:
            raise ValueError('')

        self.args = args
        # construct attention-based component to select classifier initializer for a task
        self.dict_init = torch.nn.Parameter(torch.zeros(args.num_class, self.h_dim + 1))        
        self.slf_attn = SimpleAttention(self.h_dim, self.h_dim, self.h_dim + 1, dropout=0.5)
        
    def set_dict(self, new_dict):
        if torch.cuda.is_available():
            self.dict_init.data = new_dict.data.cuda()
        else:
            self.dict_init.data = new_dict.data

    def forward(self, data_shot, data_query):
        # obtain classifier initializer based on current support data embedding
        support = self.encoder(data_shot, embedding=True)
        proto = support.reshape(self.args.shot, -1, support.shape[-1]).mean(dim=0) # N x d
        # split dict_init to get classifier and bias
        classifier_dict, bias_dict = self.dict_init.split(self.h_dim, 1)
        classifier_init = self.slf_attn(torch.unsqueeze(proto, 0), 
                                        torch.unsqueeze(classifier_dict, 0), 
                                        torch.unsqueeze(self.dict_init, 0)).squeeze(0)

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