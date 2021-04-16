import torch
import torch.nn as nn
import torch.nn.functional as F
from ops import AbsDetJacobian, loss_fn, make_pairs
from architecture import Encoder, Decoder


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
        self.mode = arg.mode
        self.background = arg.background
        self.fold_with_L_inv = arg.fold_with_L_inv

        self.n_parts = arg.n_parts + 1 if self.background else arg.n_parts
        self.encoder = Encoder(arg.n_parts, arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s, arg.depth_a,
                               arg.p_dropout, arg.hg_patch_size, arg.hg_dim, arg.hg_depth, arg.hg_heads, arg.hg_mlp_dim,
                               arg.module, arg.device, arg.background)
        self.decoder = Decoder(arg.n_features, arg.reconstr_dim, arg.n_parts,
                               arg.dec_patch_size, arg.dec_dim, arg.dec_depth, arg.dec_heads, arg.dec_mlp_dim,
                               arg.module, arg.device, arg.background)

    def forward(self, img):
        device = img.get_device()
        bn = img.shape[0]
        # Make Transformation
        input_images, ground_truth_images, mesh_stack = make_pairs(img, self.arg)
        transform_mesh = F.interpolate(mesh_stack, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh, device)

        # Send through encoder
        mu, L_inv, part_map_norm, heat_map, heat_map_norm, part_appearances = self.encoder(input_images)

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

        loss = loss_fn(bn, mu, L_inv, mu_t, stddev_t, img_reconstr,
                                                            ground_truth_images, self.l_2_scal, self.l_2_threshold,
                                                            self.L_mu, self.L_cov, self.L_rec, device,
                                                            self.background, self.fold_with_L_inv)
        if self.background:
            mu, L_inv = mu[:, :-1], L_inv[:, :-1]

        return ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, loss