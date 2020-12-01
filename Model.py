import torch
import torch.nn as nn
from architecture_ops import E, Decoder
from ops import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec, total_loss, get_heat_map, prepare_pairs, AbsDetJacobian
from transformations import ThinPlateSpline, tps_parameters, make_input_tps_param
import torch.nn.functional as F
from opt_einsum import contract

class Model(nn.Module):
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg
        self.mode = arg.mode
        self.reconstr_dim = arg.reconstr_dim
        self.n_parts = arg.n_parts
        self.n_features = arg.n_features
        self.device = arg.device
        self.depth_s = arg.depth_s
        self.depth_a = arg.depth_a
        self.residual_dim = arg.residual_dim
        self.covariance = arg.covariance
        self.L_mu = arg.L_mu
        self.L_cov = arg.L_cov
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.tps_scal = arg.tps_scal
        self.scal = arg.scal
        self.L_inv_scal = arg.L_inv_scal
        self.E_sigma = E(self.depth_s, self.n_parts, residual_dim=self.residual_dim, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, residual_dim=self.residual_dim, sigma=False)
        self.decoder = Decoder(self.n_parts, self.n_features, self.reconstr_dim)

    def forward(self, x, x_spatial_transform, x_appearance_transform, coord, vector):
        # Shape Stream
        shape_stream_parts_raw, shape_stream_parts, shape_stream_sum = self.E_sigma(x_appearance_transform)
        mu, L_inv = get_mu_and_prec(shape_stream_parts, self.device, self.L_inv_scal)
        heat_map = get_heat_map(mu, L_inv, self.device)
        # Appearance Stream
        appearance_stream_parts_raw, appearance_stream_parts, appearance_stream_sum = self.E_sigma(x_spatial_transform)
        local_features = self.E_alpha(appearance_stream_sum)
        local_part_appearances = get_local_part_appearances(local_features, appearance_stream_parts)
        # Decoder
        encoding = feat_mu_to_enc(local_part_appearances, mu, L_inv, self.device, self.covariance, self.reconstr_dim)
        reconstruction = self.decoder(encoding)
        # Loss
        loss = total_loss(x, reconstruction, shape_stream_parts, appearance_stream_parts, mu, coord, vector,
                          self.device, self.L_mu, self.L_cov, self.scal, self.l_2_scal, self.l_2_threshold)

        if self.mode == 'predict':
            return x, shape_stream_parts_raw, appearance_stream_parts_raw, reconstruction

        elif self.mode == 'train':
            return reconstruction, loss


class Model2(nn.Module):
    def __init__(self, arg):
        super(Model2, self).__init__()
        self.arg = arg
        self.bn = arg.bn
        self.mode = arg.mode
        self.n_parts = arg.n_parts
        self.n_features = arg.n_features
        self.device = arg.device
        self.depth_s = arg.depth_s
        self.depth_a = arg.depth_a
        self.residual_dim = arg.residual_dim
        self.covariance = arg.covariance
        self.L_mu = arg.L_mu
        self.L_cov = arg.L_cov
        self.tps_scal = arg.tps_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.scal = arg.scal
        self.L_inv_scal = arg.L_inv_scal
        self.E_sigma = E(self.depth_s, self.n_parts, residual_dim=self.residual_dim, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, residual_dim=self.residual_dim, sigma=False)
        self.decoder = Decoder(self.n_parts, self.n_features)

    def forward(self, x):
        # tps
        image_orig = x.repeat(2, 1, 1, 1)
        tps_param_dic = tps_parameters(image_orig.shape[0], self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector = make_input_tps_param(tps_param_dic)
        coord, vector = coord.to(self.device), vector.to(self.device)
        t_images, t_mesh = ThinPlateSpline(image_orig, coord, vector, 128, device=self.device)
        image_in, image_rec = prepare_pairs(t_images, self.arg)
        transform_mesh = F.interpolate(t_mesh, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh, self.device)

        # encoding
        part_maps, sum_part_maps = self.E_sigma(image_in)
        mu, L_inv = get_mu_and_prec(part_maps, self.device, self.L_inv_scal)
        heat_map = get_heat_map(mu, L_inv, self.device)
        raw_features = self.E_alpha(sum_part_maps)
        features = get_local_part_appearances(raw_features, part_maps)

        # transform
        integrant = (part_maps.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
        mu_t = contract('akij, alij -> akl', integrant, transform_mesh)
        transform_mesh_out_prod = contract('amij, anij -> amnij', transform_mesh, transform_mesh)
        mu_out_prod = contract('akm, akn -> akmn', mu_t, mu_t)
        stddev_t = contract('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod

        # processing
        encoding = feat_mu_to_enc(features, mu, L_inv, self.device, self.covariance)
        reconstruct_same_id = self.decoder(encoding)

        loss = nn.MSELoss()(image_rec, reconstruct_same_id)

        if self.mode == 'predict':
            return image_in, image_rec, mu, heat_map

        elif self.mode == 'train':
            return reconstruct_same_id, loss







