from collections import OrderedDict
from torch import nn


HASH = '5c427c9c'


class Flatten(nn.Module):

    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):

    """
    Helper module that stores the current tensor. Useful for accessing by name
    """

    def forward(self, x):
        return x

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class CORblock_Z(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,low_dim=128,l2norm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
                                      
        #self.groupnorm = nn.GroupNorm(32, out_channels) #This is imported from open_cl alexnet_gn    
        self.batchnorm = nn.BatchNorm2d(out_channels)                          
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        #x = self.groupnorm(x) #shown to be important for good acc
        x = self.batchnorm(x)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)  # for an easy access to this block's output
        return x

'''
Vlad edited this to have a pIT and aIT layer
can correspond to LO -> pFs
or OFA -> FFA

To make this compatible with the contrastive model, i also removed the output layer and changes the last FC to output 128
'''

def cornet_z_cl(low_dim=128):
    model = nn.Sequential(OrderedDict([
        ('V1', CORblock_Z(3, 64, kernel_size=7, stride=2)),
        ('V2', CORblock_Z(64, 128)),
        ('V4', CORblock_Z(128, 256)),
        ('pIT', CORblock_Z(256, 512)),
        ('aIT', CORblock_Z(512, 1024)),
        ('decoder', nn.Sequential(OrderedDict([
            ('avgpool', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(1024, low_dim)),
            ('l2norm', Normalize(2))
        ])))
    ]))

    # weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            #nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='relu')  #vlad changed this
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model