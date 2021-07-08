import torch.nn as nn

# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.BatchNorm2d(out_channels, momentum=0),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_block_nopool(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels, momentum=1),
        nn.ReLU()
    )

class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block_nopool(hid_dim, hid_dim),
            conv_block_nopool(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

