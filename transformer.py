import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from performer_pytorch import SelfAttention
from gsa_pytorch import GSA

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
    def __init__(self, dim, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
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
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, SelfAttention(dim, heads = heads, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, nk, n_token, use_first, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective (at least 16). Try decreasing your patch size'

        self.nk = nk
        self.patch_size = patch_size
        self.n_token = n_token
        self.use_first = use_first

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + n_token, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, n_token, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()
        self.up_conv1 = nn.ConvTranspose2d(in_channels=self.nk, out_channels=self.nk, kernel_size=2, stride=2,
                                           padding=0)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=self.nk, out_channels=self.nk, kernel_size=2, stride=2,
                                           padding=0)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=self.nk, out_channels=self.nk, kernel_size=2, stride=2,
                                           padding=0)
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(self.nk)
        self.bn2 = nn.InstanceNorm2d(self.nk)
        self.bn3 = nn.InstanceNorm2d(self.nk)


        if self.use_first:
            self.mlp_head = nn.Sequential(
                nn.Linear(self.n_token* dim, 1024),
                nn.BatchNorm1d(1024)
            )
        else:
            self.mlp_head = nn.Sequential(
                nn.Linear(num_patches * dim, 1024),
                nn.BatchNorm1d(1024),
            )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, d = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + self.nk)]
        x = self.dropout(x)

        x = self.transformer(x, mask)
        if self.use_first:
            x = self.to_cls_token(x[:, :self.n_token])
        else:
            x = self.to_cls_token(x[:, self.n_token:])
        x = x.reshape(b, -1)
        x = self.mlp_head(x)

        x = x.reshape(b, self.nk, 8, 8)
        x = self.up_conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.up_conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.up_conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


if __name__ == '__main__':
    gsa = GSA(
        dim=256,
        dim_out=64,
        dim_key=32,
        heads=8,
        rel_pos_length=256  # in paper, set to max(height, width). you can also turn this off by omitting this line
    )
    part_map = torch.randn(1, 256, 64, 64)
    out = gsa(part_map)
    print(out.shape)