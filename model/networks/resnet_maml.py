# ResNet Wide Version as in Qiao's Paper
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, stride=1, downsample=None):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if downsample is not None:
            residual = downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
def block_forward_para(x, params, base, mode, modules, downsample=False, stride=2):
    '''the forard function of BasicBlock give parametes'''
    residual = x
    out = F.conv2d(x, params[base + 'conv1.weight'], stride=stride, padding=1)
    out = F.batch_norm(out, weight=params[base + 'bn1.weight'], bias=params[base + 'bn1.bias'],
                       running_mean=modules['bn1'].running_mean,
                       running_var=modules['bn1'].running_var, training = mode)
    out = F.relu(out)
    out = F.conv2d(out, params[base + 'conv2.weight'], stride=1, padding=1)
    out = F.batch_norm(out, weight=params[base + 'bn2.weight'], bias=params[base + 'bn2.bias'],
                           running_mean=modules['bn2'].running_mean,
                           running_var=modules['bn2'].running_var, training = mode)        
    
    if downsample is True:
        residual = F.conv2d(x, params[base + 'downsample.0.weight'], stride=2)
        residual = F.batch_norm(residual, weight=params[base + 'downsample.1.weight'], 
                                bias=params[base + 'downsample.1.bias'],
                                running_mean=modules['downsample']._modules['1'].running_mean,
                                running_var=modules['downsample']._modules['1'].running_var, training = mode)
    out += residual
    out = F.relu(out)
    return out

def forward_layer(x, params, base, mode, modules, blocks):
    # forward of a layer given parameters
    x = block_forward_para(x, params, base + '.0.', mode, modules['0']._modules, True, stride=2)
    for i in range(1, blocks):
        x = block_forward_para(x, params, base + '.{0}.'.format(i), mode, modules[str(i)]._modules, False,  stride=1)

    return x


class ResNet(nn.Module):

    def __init__(self, n_class, block=BasicBlock, layers=[4,4,4]):
        super(ResNet, self).__init__()
        self.is_training = True
        self.n_class = n_class        
        cfg = [160, 320, 640]
        self.inplanes = iChannels = int(cfg[0]/2)
        
        self.conv1 = nn.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(10, stride=1)
        # 512 * block.expansion
        self.FC = nn.Linear(cfg[2], n_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, params = None, embedding = False):
        if params is None:
            params = OrderedDict(self.named_parameters())
                    
        x = F.conv2d(x, params['conv1.weight'], bias=params['conv1.bias'], stride=1, padding=1)
        x = F.batch_norm(x, weight=params['bn1.weight'], bias=params['bn1.bias'],
                         running_mean=self._modules['bn1'].running_mean,
                         running_var=self._modules['bn1'].running_var, training = self.is_training)
        x = self.relu(x)
        
        x = forward_layer(x, params, 'layer1', self.is_training, self._modules['layer1']._modules, 4)
        x = forward_layer(x, params, 'layer2', self.is_training, self._modules['layer2']._modules, 4)
        x = forward_layer(x, params, 'layer3', self.is_training, self._modules['layer3']._modules, 4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        if embedding:
            return x
        else:
            # Apply Linear Layer
            logits = F.linear(x, weight=params['FC.weight'], bias=params['FC.bias'])
            return logits
