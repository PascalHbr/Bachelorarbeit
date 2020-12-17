import torch
import torch.nn as nn
from transformer import ViT, ViT2

def softmax(logit_map):
    bn, kn, h, w = logit_map.shape
    map_norm = nn.Softmax(dim=2)(logit_map.reshape(bn, kn, -1)).reshape(bn, kn, h, w)
    return map_norm


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True):
        super(Conv, self).__init__()
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


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, bn=False, relu=False)
        self.bn2 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, bn=False, relu=False)
        self.bn3 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, bn=False, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, bn=False, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f):
        super(Hourglass, self).__init__()
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, f)
        self.n = n
        # Recursive hourglass
        if self.n > 0:
            self.low2 = Hourglass(n - 1, f)
        else:
            self.low2 = Residual(f, f)
        self.low3 = Residual(f, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class E(nn.Module):
    def __init__(self, depth, n_feature, residual_dim, p_dropout, sigma=True, reconstr_dim=256):
        super(E, self).__init__()
        self.sigma = sigma
        self.dropout = nn.Dropout(p_dropout)
        self.reconstr_dim = reconstr_dim
        self.hg = Hourglass(depth, residual_dim)  # depth 4 has bottleneck of 4x4
        self.out = Conv(residual_dim, residual_dim, kernel_size=1, stride=1, bn=True, relu=True)
        self.feature = Conv(residual_dim, n_feature, kernel_size=1, stride=1, bn=False, relu=False)
        # Preprocessing
        if self.sigma:
            if self.reconstr_dim == 128:
                self.preprocess_sigma = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                      Residual(64, residual_dim)
                                                      )
            elif self.reconstr_dim == 256:
                self.preprocess_sigma = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                      Residual(64, 128),
                                                      nn.MaxPool2d(2, 2),
                                                      Residual(128, 128),
                                                      Residual(128, residual_dim)
                                                      )
            self.map_transform = Conv(n_feature, residual_dim, 1, 1, bn=False, relu=False)  # channels for addition must be increased
        if not self.sigma:
            self.preprocess_alpha = Conv(2 * residual_dim, residual_dim, 1, 1, bn=True, relu=True)  # for stack

    def forward(self, x):
        if self.sigma:
            x = self.preprocess_sigma(x)
        # else:
        #     x = self.preprocess_alpha(x) # Try concatenation instead of sum
        out = self.hg(x)
        out = self.dropout(out)
        out = self.out(out)
        # Get Normalized Feature Maps for E_sigma
        feature_map = self.feature(out)
        if self.sigma:
            map_normalized = softmax(feature_map)
            map_transformed = self.map_transform(map_normalized)
            # stack = torch.cat((map_transformed, x), dim=1) # Try concatenation instead of sum
            stack = map_transformed + x
            return feature_map, map_normalized, stack
        else:
            return feature_map


class E_transformer(nn.Module):
    def __init__(self, depth, n_feature, residual_dim, p_dropout, sigma=True, reconstr_dim=256):
        super(E_transformer, self).__init__()
        self.sigma = sigma
        self.n_feature = n_feature
        self.dropout = nn.Dropout(p_dropout)
        self.reconstr_dim = reconstr_dim
        self.hg = Hourglass(depth, residual_dim)  # depth 4 has bottleneck of 4x4
        self.VT = ViT(
            image_size=256,
            patch_size=32,
            dim=1024,
            depth=16,
            heads=8,
            mlp_dim=128,
            dropout=0.1,
            emb_dropout=0.1,
            nk=n_feature,
            sigma=self.sigma
        )

        self.out = Conv(residual_dim, residual_dim, kernel_size=1, stride=1, bn=True, relu=True)
        self.feature = Conv(n_feature, n_feature, kernel_size=1, stride=1, bn=False, relu=False)
        # Preprocessing
        if self.sigma:
            if self.reconstr_dim == 128:
                self.preprocess_sigma = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                      Residual(64, residual_dim)
                                                      )
            elif self.reconstr_dim == 256:
                self.preprocess_sigma = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                      Residual(64, 128),
                                                      nn.MaxPool2d(2, 2),
                                                      Residual(128, 128),
                                                      Residual(128, residual_dim)
                                                      )
            self.map_transform = Conv(n_feature, residual_dim, 1, 1, bn=False, relu=False)  # channels for addition must be increased


    def forward(self, x):
        if self.sigma:
            x_stack = self.preprocess_sigma(x)
        out = self.VT(x)
        # Get Normalized Feature Maps for E_sigma
        feature_map = self.feature(out)
        if self.sigma:
            map_normalized = softmax(feature_map)
            map_transformed = self.map_transform(map_normalized)
            stack = map_transformed + x_stack
            return feature_map, map_normalized, stack
        else:
            return feature_map


class Nccuc(nn.Module):
    def __init__(self, in_channels, filters):
        super(Nccuc, self).__init__()
        self.in_channels = in_channels
        self.n_filters = filters
        self.down_Conv = Conv(inp_dim=in_channels, out_dim=filters[0], kernel_size=3, stride=1, bn=True, relu=True)
        self.up_Conv = nn.ConvTranspose2d(in_channels=filters[0], out_channels=filters[1], kernel_size=4, stride=2,
                                          padding=1)

    def forward(self, input_A, input_B):
        down_conv = self.down_Conv(input_A)
        up_conv = self.up_Conv(down_conv)
        out = torch.cat((up_conv, input_B), dim=1)
        return out


class Decoder(nn.Module):
    def __init__(self, nk, nf, reconstr_dim, n_c=3):
        super(Decoder, self).__init__()
        self.reconstr_dim = reconstr_dim
        self.out_channels = n_c
        self.conv1 = Nccuc(nf + 2, [512, 512])  # 8
        self.conv2 = Nccuc(nf + 4 + 512, [512, 256])  # 16
        self.conv3 = Nccuc(nf + nk + 256, [256, 256])  # 32

        if reconstr_dim == 128:
            self.conv4 = Nccuc(nk + 256, [256, 128])  # 64
            self.conv5 = Nccuc(nk + 128, [128, 64])   # 128
            self.conv6 = Conv(nk + 64, self.out_channels, kernel_size=5, stride=1, bn=False, relu=False)

        if reconstr_dim == 256:
            self.conv4 = Nccuc(nk + 256, [256, 128])  # 64
            self.conv5 = Nccuc(nk + 128, [128, 128])    # 128
            self.conv6 = Nccuc(nk + 128, [128, 64])     # 256
            self.conv7 = Conv(nk + 64, self.out_channels, kernel_size=5, stride=1, bn=False, relu=False)

    def forward(self, encoding_list):
        conv1 = self.conv1(encoding_list[-1], encoding_list[-2])
        conv2 = self.conv2(conv1, encoding_list[-3])
        conv3 = self.conv3(conv2, encoding_list[-4])
        conv4 = self.conv4(conv3, encoding_list[-5])
        conv5 = self.conv5(conv4, encoding_list[-6])
        if self.reconstr_dim == 128:
            conv6 = self.conv6(conv5)
            out = torch.sigmoid(conv6)
        if self.reconstr_dim == 256:
            conv6 = self.conv6(conv5, encoding_list[-7])
            conv7 = self.conv7(conv6)
            out = torch.sigmoid(conv7)
        return out