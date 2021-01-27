import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from opt_einsum import contract
from architecture import E, Decoder, E_transformer
from ops import prepare_pairs, AbsDetJacobian, feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec, loss_fn, augm
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
from ops import get_mask


class Model(nn.Module):
    def __init__(self, arg):
        super(Model, self).__init__()
        self.arg = arg
        self.mode = arg.mode
        self.bn = arg.batch_size
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
        self.L_rec = arg.L_rec
        self.L_sep = arg.L_sep
        self.sig_sep = arg.sig_sep
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.map_threshold = arg.map_threshold
        self.tps_scal = arg.tps_scal
        self.scal = arg.scal
        self.L_inv_scal = arg.L_inv_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.static = arg.static
        self.fold_with_shape = arg.fold_with_shape
        # self.dlab = models.segmentation.deeplabv3_resnet50(pretrained=1).eval()
        self.E_sigma = E(self.depth_s, self.n_parts, self.residual_dim, self.p_dropout,
                         sigma=True, reconstr_dim=arg.reconstr_dim)
        # self.E_sigma = E_transformer(self.arg, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, self.residual_dim, self.p_dropout,
                         sigma=False, reconstr_dim=arg.reconstr_dim)
        # self.E_alpha = E_transformer(self.arg, sigma=False)
        self.decoder = Decoder(self.n_parts + 1, self.n_features, self.reconstr_dim)


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
        image_in, image_rec = prepare_pairs(t_images, self.arg, self.device)
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
        encoding = feat_mu_to_enc(features, mu, L_inv, self.device, self.covariance, self.reconstr_dim, self.static)
        reconstruct_same_id = self.decoder(encoding)

        total_loss, rec_loss, transform_loss, precision_loss = loss_fn(batch_size, mu, L_inv, mu_t, stddev_t,
                                                                       reconstruct_same_id, image_rec, self.fold_with_shape,
                                                                       self.l_2_scal, self.l_2_threshold, self.L_mu, self.L_cov,
                                                                       self.L_rec, self.L_sep, self.sig_sep,
                                                                       self.device)

        # norms
        original_part_maps_raw, original_part_maps_norm, original_sum_part_maps = self.E_sigma(x)
        mu_original, L_inv_original = get_mu_and_prec(original_part_maps_norm, self.device, self.L_inv_scal)

        if self.mode == 'predict':
            original_part_maps_raw, original_part_maps_norm, original_sum_part_maps = self.E_sigma(x)
            return original_part_maps_raw, mu_original[:, :-1], image_rec, part_maps_raw, part_maps_raw, reconstruct_same_id

        elif self.mode == 'train':
            return image_rec, reconstruct_same_id, total_loss, rec_loss, transform_loss, precision_loss, mu[:, :-1], L_inv[:, :-1], mu_original[:, :-1]


class Model2(nn.Module):
    def __init__(self, arg):
        super(Model2, self).__init__()
        self.arg = arg
        self.mode = arg.mode
        self.bn = arg.batch_size
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
        self.L_rec = arg.L_rec
        self.L_sep = arg.L_sep
        self.sig_sep = arg.sig_sep
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.map_threshold = arg.map_threshold
        self.tps_scal = arg.tps_scal
        self.scal = arg.scal
        self.L_inv_scal = arg.L_inv_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.static = arg.static
        self.fold_with_shape = arg.fold_with_shape
        self.E_sigma = E(self.depth_s, self.n_parts, self.residual_dim, self.p_dropout,
                         sigma=True, reconstr_dim=arg.reconstr_dim)
        self.E_alpha = E(self.depth_a, self.n_features, self.residual_dim, self.p_dropout,
                         sigma=False, reconstr_dim=arg.reconstr_dim)
        self.decoder = Decoder(self.n_parts + 1, self.n_features, self.reconstr_dim)


    def forward(self, x):
        batch_size = x.shape[0]
        # tps
        tps_param_dic = tps_parameters(batch_size, self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector, rot_mat = make_input_tps_param(tps_param_dic)
        coord, vector, rot_mat = coord.to(self.device), vector.to(self.device), rot_mat.to(self.device)
        x_TPS, t_mesh = ThinPlateSpline(x, coord, vector, self.reconstr_dim, device=self.device)
        transform_mesh = F.interpolate(t_mesh, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh, self.device)
        x_augm = augm(x, self.arg, self.device)

        # Shape Stream
        part_maps_raw_shape, part_maps_norm_shape, sum_part_maps_shape = self.E_sigma(x_augm)
        mu_shape, L_inv_shape = get_mu_and_prec(part_maps_norm_shape, self.device, self.L_inv_scal)

        integrant = (part_maps_norm_shape.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
        mu_shape_t = contract('akij, alij -> akl', integrant, transform_mesh)
        transform_mesh_out_prod = contract('amij, anij -> amnij', transform_mesh, transform_mesh)
        mu_out_prod_shape = contract('akm, akn -> akmn', mu_shape_t, mu_shape_t)
        stddev_t_shape = contract('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod_shape

        # Appearance Stream
        part_maps_raw_app, part_maps_norm_app, sum_part_maps_app = self.E_sigma(x_TPS)
        mu_app, L_inv_app = get_mu_and_prec(part_maps_norm_app, self.device, self.L_inv_scal)
        raw_features = self.E_alpha(sum_part_maps_app)
        features = get_local_part_appearances(raw_features, part_maps_norm_app)

        y_t = torch.linspace(-1., 1., 64, device=self.device).reshape(64, 1).repeat(1, 64).unsqueeze(-1)
        x_t = torch.linspace(-1., 1., 64, device=self.device).reshape(1, 64).repeat(64, 1).unsqueeze(-1)
        meshgrid = torch.cat((y_t, x_t), dim=-1).unsqueeze(0).repeat(batch_size, 1, 1, 1).permute(0, 3, 1, 2)
        volume_mesh_app = AbsDetJacobian(meshgrid, self.device)
        integrant_app = (part_maps_norm_app.unsqueeze(-1) * volume_mesh_app.unsqueeze(-1)).squeeze()
        transform_mesh_out_prod_app = contract('amij, anij -> amnij', meshgrid, meshgrid)
        mu_out_prod_app = contract('akm, akn -> akmn', mu_app, mu_app)
        stddev_t_app = contract('akij, amnij -> akmn', integrant_app, transform_mesh_out_prod_app) - mu_out_prod_app

        # processing
        encoding = feat_mu_to_enc(features, mu_shape, L_inv_shape, self.device, self.covariance, self.reconstr_dim, self.static)
        reconstruction = self.decoder(encoding)

        mu_t = torch.cat((mu_shape_t, mu_app), dim=0)
        stddev_t = torch.cat((stddev_t_shape, stddev_t_app), dim=0)

        total_loss, rec_loss, transform_loss, precision_loss = loss_fn(batch_size, mu_shape, L_inv_shape, mu_t, stddev_t,
                                                                       reconstruction, x, self.fold_with_shape,
                                                                       self.l_2_scal, self.l_2_threshold, self.L_mu, self.L_cov,
                                                                       self.L_rec, self.L_sep, self.sig_sep,
                                                                       self.device)

        # norms
        original_part_maps_raw, original_part_maps_norm, original_sum_part_maps = self.E_sigma(x)
        mu_original, L_inv_original = get_mu_and_prec(original_part_maps_norm, self.device, self.L_inv_scal)
        image_in = torch.cat((x_augm, x_TPS), dim=0)

        if self.mode == 'predict':
            original_part_maps_raw, original_part_maps_norm, original_sum_part_maps = self.E_sigma(x)
            part_maps_raw = torch.cat((part_maps_raw_shape, part_maps_raw_app), dim=0)
            return original_part_maps_raw, mu_original[:, :-1], image_in, part_maps_raw, part_maps_raw, reconstruction

        elif self.mode == 'train':
            return image_in, reconstruction, total_loss, rec_loss, transform_loss, precision_loss, mu_shape[:, :-1], L_inv_shape[:, :-1], mu_original[:, :-1]
