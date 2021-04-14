import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from performer_pytorch import SelfAttention


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

        # self.initialize_weights()

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


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, bn=False, relu=False)
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv(in_channels, out_channels, kernel_size=3, stride=1, bn=False, relu=False)
        self.bn2 = nn.InstanceNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

MIN_NUM_PATCHES = 16

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, nk, pool = 'mean', channels=3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.patch_size = patch_size
        self.map_size = int(num_patches**0.5)
        self.nk = nk

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.conv = Conv(dim, channels, kernel_size=3, stride=1, relu=True, bn=True)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(channels)
        self.bn2 = nn.InstanceNorm2d(channels)
        self.bn3 = nn.InstanceNorm2d(channels)
        self.up_Conv1 = nn.Sequential(ResidualBlock(channels, channels),
                                      nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4,
                                                         stride=2, padding=1),
                                      self.bn1,
                                      self.relu)

        self.up_Conv2 = nn.Sequential(ResidualBlock(channels, channels),
                                      nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4,
                                                         stride=2, padding=1),
                                      self.bn2,
                                      self.relu)

        self.up_Conv3 = nn.Sequential(ResidualBlock(channels, channels),
                                      nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4,
                                                         stride=2, padding=1),
                                      self.bn3,
                                      self.relu)

        self.to_partmap = Conv(channels, nk, kernel_size=3, stride=1, bn=False, relu=False)


    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, c = x.shape

        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = x.permute(0, 2, 1).reshape(b, c, self.map_size, self.map_size)
        x = self.conv(x)

        if self.map_size < 64:
            x = self.up_Conv1(x)
        if self.map_size < 32:
            x = self.up_Conv2(x)
        if self.map_size < 16:
            x = self.up_Conv3(x)

        x = self.to_partmap(x)

        return x

if __name__ == '__main__':
    VT = ViT(
        image_size=64,
        patch_size=4,
        dim=256,
        depth=8,
        heads=8,
        mlp_dim=256,
        dropout=0.1,
        channels=256,
        emb_dropout=0.1,
        nk=17
    )

    img = torch.randn(8, 256, 64, 64)
    out = VT(img)
    print(out.shape)