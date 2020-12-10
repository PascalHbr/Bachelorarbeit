import torch
import torch.nn as nn
import torch.nn.functional as F
from opt_einsum import contract
from architecture_ops import E, Decoder
from ops import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec
from ops2 import prepare_pairs, AbsDetJacobian, loss_fn
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline

class Model2(nn.Module):
    def __init__(self, arg):
        super(Model2, self).__init__()
        self.arg = arg
        self.mode = arg.mode
        self.bn = arg.bn
        self.reconstr_dim = arg.reconstr_dim
        self.n_parts = arg.n_parts
        self.n_features = arg.n_features
        self.device = arg.device
        self.depth_s = arg.depth_s
        self.depth_a = arg.depth_a
        self.p_dropout = arg.p_dropout
        self.residual_dim = arg.residual_dim
        self.covariance = arg.covariance
        self.L_mu = arg.L_mu
        self.L_cov = arg.L_cov
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.tps_scal = arg.tps_scal
        self.scal = arg.scal
        self.L_inv_scal = arg.L_inv_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.fold_with_shape = arg.fold_with_shape
        self.E_sigma = E(self.depth_s, self.n_parts, self.residual_dim, self.p_dropout, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, self.residual_dim, self.p_dropout, sigma=False)
        self.decoder = Decoder(self.n_parts, self.n_features, self.reconstr_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        batch_size2 = 2 * x.shape[0]
        # tps
        image_orig = x.repeat(2, 1, 1, 1)
        tps_param_dic = tps_parameters(batch_size2, self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector = make_input_tps_param(tps_param_dic)
        coord, vector = coord.to(self.device), vector.to(self.device)
        t_images, t_mesh = ThinPlateSpline(image_orig, coord, vector, self.reconstr_dim, device=self.device)
        image_in, image_rec = prepare_pairs(t_images, self.arg)
        transform_mesh = F.interpolate(t_mesh, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh, self.device)

        # encoding
        part_maps_raw, part_maps_norm, sum_part_maps = self.E_sigma(image_in)
        mu, L_inv = get_mu_and_prec(part_maps_norm, self.device, self.L_inv_scal)
        raw_features = self.E_alpha(sum_part_maps)
        features = get_local_part_appearances(raw_features, part_maps_norm)

        # transform
        integrant = (part_maps_norm.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
        mu_t = contract('akij, alij -> akl', integrant, transform_mesh)
        transform_mesh_out_prod = contract('amij, anij -> amnij', transform_mesh, transform_mesh)
        mu_out_prod = contract('akm, akn -> akmn', mu_t, mu_t)
        stddev_t = contract('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod

        # processing
        encoding = feat_mu_to_enc(features, mu, L_inv, self.device, self.covariance, self.reconstr_dim)
        reconstruct_same_id = self.decoder(encoding)

        total_loss, rec_loss, transform_loss, precision_loss = loss_fn(batch_size, mu, L_inv, mu_t, stddev_t,
                                                                       reconstruct_same_id, image_rec, self.fold_with_shape,
                                                                       self.l_2_scal, self.l_2_threshold, self.L_mu, self.L_cov,
                                                                       self.device)

        if self.mode == 'predict':
            original_part_maps_raw, original_part_maps_norm, original_sum_part_maps = self.E_sigma(x)
            return original_part_maps_raw, image_rec, part_maps_raw, part_maps_raw, reconstruct_same_id

        elif self.mode == 'train':
            return image_rec, reconstruct_same_id, total_loss, rec_loss, transform_loss, precision_loss, mu, L_inv