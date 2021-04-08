import torch
import torch.nn as nn
from transformer import ViT
import torch.nn.functional as F
from ops import get_heat_map, AbsDetJacobian, loss_fn, feat_mu_to_enc, make_pairs, rotation_mat, softmax, get_mu
from architecture import Decoder as Decoder_old

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


class PreProcessor(nn.Module):
    def __init__(self, residual_dim, reconstr_dim):
        super(PreProcessor, self).__init__()
        if reconstr_dim == 128:
            self.preprocess = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                  Residual(64, 128),
                                                  Residual(128, 128),
                                                  Residual(128, residual_dim),
                                                  )
        elif reconstr_dim == 256:
            self.preprocess = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                  Residual(64, 128),
                                                  nn.MaxPool2d(2, 2),
                                                  Residual(128, 128),
                                                  Residual(128, residual_dim),
                                                  )
    def forward(self, img):
        img_preprocess = self.preprocess(img)

        return img_preprocess


class Encoder(nn.Module):
    def __init__(self, n_parts, n_features, residual_dim, reconstr_dim, depth_s, depth_a, p_dropout,
                 hg_patch_size, hg_dim, hg_depth, hg_heads, hg_mlp_dim, module, device):
        super(Encoder, self).__init__()
        self.preprocessor = PreProcessor(residual_dim, reconstr_dim)
        self.k = n_parts
        self.device = device
        self.module = module

        self.sigmoid = nn.Sigmoid()
        self.map_transform = Conv(self.k, residual_dim, 3, 1, bn=False, relu=False)

        # Layer to predict L_inv
        self.L_inv = nn.Conv2d(in_channels=self.k, out_channels=2 * self.k, kernel_size=64, groups=self.k)
        self.bn = nn.BatchNorm2d(2 * self.k)

        # Hourglass Shape
        if self.module in [1, 3]:
            self.hg_shape = Hourglass(depth_s, residual_dim)
            self.dropout = nn.Dropout(p_dropout)
            self.out = Conv(residual_dim, residual_dim, kernel_size=3, stride=1, bn=True, relu=True)
            self.to_parts = Conv(residual_dim, self.k, kernel_size=3, stride=1, bn=False, relu=False)

            # Hourglass Appearance
            self.hg_appearance = Hourglass(depth_a, residual_dim)
            self.to_features = Conv(residual_dim, n_features, kernel_size=1, stride=1, bn=False, relu=False)

        # Transformer Shape
        if self.module in [2, 4]:
            self.conv1 = Conv(residual_dim, residual_dim, kernel_size=3, stride=1, bn=True, relu=True)
            self.vit_shape = ViT(
               image_size=64,
               patch_size=hg_patch_size,
               dim=hg_dim,
               depth=hg_depth,
               heads=hg_heads,
               mlp_dim=hg_mlp_dim,
               dropout=0.1,
               channels=256,
               emb_dropout=0.1,
               nk=self.k
               )

            self.vit_appearance = ViT(
                image_size=64,
                patch_size=hg_patch_size,
                dim=hg_dim,
                depth=hg_depth,
                heads=hg_heads,
                mlp_dim=hg_mlp_dim,
                dropout=0.1,
                channels=256,
                emb_dropout=0.1,
                nk=n_features
                )

    def forward(self, img):
        bn = img.shape[0]
        img_preprocessed = self.preprocessor(img)

        # Shape Representation with HG
        if self.module in [1, 3]:
            img_shape = self.hg_shape(img_preprocessed)
            img_shape = self.dropout(img_shape)
            img_shape = self.out(img_shape)
            feature_map = self.to_parts(img_shape)

        # Shape Representation with ViT
        if self.module in [2, 4]:
            img_shape = self.conv1(img_preprocessed)
            feature_map = self.vit_shape(img_shape)

        # Get mu
        map_normalized = softmax(feature_map)
        mu = get_mu(map_normalized, self.device)

        # Get Stack for Appearance Hourglass
        map_transformed = self.map_transform(map_normalized)
        stack = map_transformed + img_preprocessed

        # Predict L_inv
        L_inv = self.L_inv(feature_map)
        L_inv = self.sigmoid(self.bn(L_inv)).reshape(bn, self.k, 2)
        rot, scal = 2 * 3.141 * L_inv[:, :, 0].reshape(-1), 20 * L_inv[:, :, 1].reshape(-1)
        scal_matrix = torch.cat([torch.tensor([[scal[i], 0.], [0., 0.]], device=self.device).unsqueeze(0) for i in range(scal.shape[0])], 0).reshape(bn, self.k, 2, 2)
        rot_mat = torch.cat([rotation_mat(rot[i].reshape(-1)).unsqueeze(0) for i in range(rot.shape[0])], 0).reshape(bn, self.k, 2, 2)
        L_inv = torch.tensor([[30., 0.], [0., 30.]], device=self.device).unsqueeze(0).unsqueeze(0).repeat(bn, self.k, 1, 1) - \
               scal_matrix
        L_inv = rot_mat @ L_inv @ rot_mat.transpose(2, 3)

        # Make Heatmap
        heat_map = get_heat_map(mu, L_inv, self.device, self.dim)
        norm = torch.sum(heat_map, 1, keepdim=True) + 1
        heat_map_norm = heat_map / norm

        # Get Appearance Representation with HG
        if self.module in [1, 3]:
            img_app = self.hg_appearance(stack)
            raw_features = self.to_features(img_app)

        # Get Appearance Representation with ViT
        if self.module in [2, 4]:
            raw_features = self.vit_appearance(stack)

        # Get Localized Part Appearances
        part_appearances = torch.einsum('bfij, bkij -> bkf', raw_features, heat_map_norm)

        return mu, L_inv, map_normalized, heat_map_norm, part_appearances


class Decoder(nn.Module):
    def __init__(self, n_features, reconstr_dim, nk,
                 dec_patch_size, dec_dim, dec_depth, dec_heads, dec_mlp_dim, module, device):
        super(Decoder, self).__init__()
        self.k = nk
        self.device = device
        self.reconstr_dim = reconstr_dim
        self.module = module

        # Choose original Decoder
        if self.module in [1, 2]:
            self.decoder_old = Decoder_old(self.k, n_features, reconstr_dim)

        # Choose ViT Decoder
        if self.module in [3, 4]:
            self.vit_decoder = ViT(
                image_size=64,
                patch_size=dec_patch_size,
                dim=dec_dim,
                depth=dec_depth,
                heads=dec_heads,
                mlp_dim=dec_mlp_dim,
                dropout=0.1,
                channels=n_features,
                emb_dropout=0.1,
                nk=256
            )
        self.relu = nn.ReLU()
        self.bn1 = nn.InstanceNorm2d(128)
        self.bn2 = nn.InstanceNorm2d(64)
        self.up_Conv1 = nn.Sequential(
                                      nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
                                                         stride=2, padding=1),
                                      self.bn1,
                                      self.relu)

        self.up_Conv2 = nn.Sequential(
                                      nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
                                                         stride=2, padding=1),
                                      self.bn2,
                                      self.relu)
        if self.reconstr_dim == 256:
            self.to_rgb = Conv(64, 3, kernel_size=3, stride=1, bn=False, relu=False)
        else:
            self.to_rgb = Conv(128, 3, kernel_size=3, stride=1, bn=False, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, heat_map_norm, part_appearances, mu, L_inv):
        # Use Original Decoder
        if self.module in [1, 2]:
            encoding = feat_mu_to_enc(part_appearances, mu, L_inv, self.device, self.reconstr_dim)
            reconstruction = self.decoder_old(encoding)

        # Use ViT
        if self.module in [3, 4]:
            encoding = torch.einsum('bkij, bkn -> bnij', heat_map_norm, part_appearances)
            out = self.vit_decoder(encoding)
            out = self.up_Conv1(out)
            if self.reconstr_dim == 256:
                out = self.up_Conv2(out)
            reconstruction = self.sigmoid(self.to_rgb(out))

        return reconstruction


class Model(nn.Module):
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg
        self.device = arg.device
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.L_mu = arg.L_mu
        self.L_cov = arg.L_cov
        self.L_rec = arg.L_rec
        self.encoder = Encoder(arg.n_parts, arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s, arg.depth_a,
                               arg.p_dropout, arg.hg_patch_size, arg.hg_dim, arg.hg_depth, arg.hg_heads, arg.hg_mlp_dim,
                               arg.module, arg.device)
        self.decoder = Decoder(arg.n_features, arg.reconstr_dim, arg.n_parts,
                               arg.dec_patch_size, arg.dec_dim, arg.dec_depth, arg.dec_heads, arg.dec_mlp_dim,
                               arg.module, arg.device)

    def forward(self, img):
        if self.mode == 'train':
            bn = img.shape[0]
            # Make Transformation
            input_images, ground_truth_images, mesh_stack = make_pairs(img, self.arg)
            transform_mesh = F.interpolate(mesh_stack, size=64)
            volume_mesh = AbsDetJacobian(transform_mesh, self.device)

            # Send through encoder
            mu, L_inv, part_map_norm, heat_map_norm, part_appearances = self.encoder(input_images)

            # Swap part appearances
            part_appearances_swap = torch.cat([part_appearances[bn:], part_appearances[:bn]], dim=0)

            # Send through decoder
            img_reconstr = self.decoder(heat_map_norm, part_appearances_swap, mu, L_inv)

            # Calculate Loss
            integrant = (part_map_norm.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
            integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
            mu_t = torch.einsum('akij, alij -> akl', integrant, transform_mesh)
            transform_mesh_out_prod = torch.einsum('amij, anij -> amnij', transform_mesh, transform_mesh)
            mu_out_prod = torch.einsum('akm, akn -> akmn', mu_t, mu_t)
            stddev_t = torch.einsum('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod

            total_loss, rec_loss, transform_loss, precision_loss = loss_fn(bn, mu, L_inv, mu_t, stddev_t, img_reconstr,
                                                                ground_truth_images, self.l_2_scal, self.l_2_threshold,
                                                                self.L_mu, self.L_cov, self.L_rec, self.device)

            return ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map_norm, total_loss

        elif self.mode == 'predict':
            mu, L_inv, part_map_norm, heat_map_norm, part_appearances = self.encoder(img)
            return mu