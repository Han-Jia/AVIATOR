import torch.nn as nn
from model.utils import euclidean_metric

class ProtoNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.model_type == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.model_type == 'ResNet':
            from model.networks.resnet34 import resnet34 as ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')
        if hasattr(args, 'train_way'):
            self.way = args.train_way
        else:
            self.way = args.way

    def forward(self, data_shot, data_query, way = None):
        if way is None:
            way = self.way
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, way, -1).mean(dim=0)
        logits = euclidean_metric(self.encoder(data_query), proto) / self.args.temperature
        return logits