import torch.nn as nn
from architecture_ops import E, Decoder
from ops import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec, total_loss


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
        self.fold_with_shape = arg.fold_with_shape
        self.E_sigma = E(self.depth_s, self.n_parts, self.residual_dim, self.p_dropout, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, self.residual_dim, self.p_dropout, sigma=False)
        self.decoder = Decoder(self.n_parts, self.n_features, self.reconstr_dim)

    def forward(self, x, x_spatial_transform, x_appearance_transform, coord, vector):
        # Shape Stream
        shape_stream_parts_raw, shape_stream_parts_norm, shape_stream_sum = self.E_sigma(x_appearance_transform)
        mu, L_inv = get_mu_and_prec(shape_stream_parts_norm, self.device, self.L_inv_scal)
        # Appearance Stream
        appearance_stream_parts_raw, appearance_stream_parts_norm, appearance_stream_sum = self.E_sigma(x_spatial_transform)
        local_features = self.E_alpha(appearance_stream_sum)
        local_part_appearances = get_local_part_appearances(local_features, appearance_stream_parts_norm)
        # Decoder
        encoding = feat_mu_to_enc(local_part_appearances, mu, L_inv, self.device, self.covariance, self.reconstr_dim)
        reconstruction = self.decoder(encoding)
        # Loss
        loss, rec_loss, equiv_loss = total_loss(x, reconstruction, shape_stream_parts_raw, appearance_stream_parts_norm,
                                                mu, L_inv, coord, vector, self.device, self.L_mu, self.L_cov, self.scal,
                                                self.l_2_scal, self.l_2_threshold, self.fold_with_shape)

        if self.mode == 'predict':
            return x, shape_stream_parts_raw, appearance_stream_parts_raw, reconstruction

        elif self.mode == 'train':
            return reconstruction, loss, rec_loss, equiv_loss, mu, L_inv
