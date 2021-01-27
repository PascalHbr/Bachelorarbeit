import torch
from gsa_pytorch import GSA
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.LeakyReLU()
        if bn:
            self.bn = nn.InstanceNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


class GSA_Transformer(nn.Module):
    def __init__(self, n_features, dim, dim_out, dim_key, heads, rel_pos_length):
        super(GSA_Transformer, self).__init__()
        self.gsa = GSA(
              dim = dim,
              dim_out = dim_out,
              dim_key = dim_key,
              heads = heads,
              rel_pos_length = rel_pos_length  # in paper, set to max(height, width). you can also turn this off by omitting this line
              )
        self.to_features = Conv(dim_out, n_features, 3, 1, relu=False, bn=False)

    def forward(self, x):
        x = self.gsa(x)
        x = self.to_features(x)
        return x

if __name__ == "__main__":
    gsa = GSA_Transformer(
              dim = 256,
              dim_out = 256,
              dim_key = 32,
              heads = 8,
              rel_pos_length = 64  # in paper, set to max(height, width). you can also turn this off by omitting this line
              )
    x = torch.randn(1, 256, 64, 64)
    out = gsa(x)
    print(out.shape)