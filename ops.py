import torch
import torch.nn.functional as F
from torchvision import models
import torchvision.transforms as T
import kornia.augmentation as K
from opt_einsum import contract
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline


def AbsDetJacobian(batch_meshgrid, device):
    """
        :param device:
        :param batch_meshgrid: takes meshgrid tensor of dim [bn, 2, h, w] (conceptually meshgrid represents a two dimensional function f = [fx, fy] on [bn, h, w] )
        :return: returns Abs det of  Jacobian of f of dim [bn, 1, h, w]
        """
    y_c = batch_meshgrid[:, 0, :, :].unsqueeze(1)
    x_c = batch_meshgrid[:, 1, :, :].unsqueeze(1)
    sobel_x_filter = 1 / 4 * torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float, device=device).reshape(1, 1, 3, 3)
    sobel_y_filter = sobel_x_filter.permute(0, 1, 3, 2)

    filtered_y_y = F.conv2d(y_c, sobel_y_filter, stride=1, padding=1)
    filtered_y_x = F.conv2d(y_c, sobel_x_filter, stride=1, padding=1)
    filtered_x_y = F.conv2d(x_c, sobel_y_filter, stride=1, padding=1)
    filtered_x_x = F.conv2d(x_c, sobel_x_filter, stride=1, padding=1)

    Det = torch.abs(filtered_y_y * filtered_x_x - filtered_y_x * filtered_x_y)

    return Det


def augm(t, arg, device):
    t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(t)
    random_tensor = 1. + torch.rand(size=[1], dtype=t.dtype, device=device)
    binary_tensor = torch.floor(random_tensor)
    random_tensor, binary_tensor = random_tensor, binary_tensor

    augmented = binary_tensor * t + (1 - binary_tensor) * (1 - t)
    return augmented


def make_pairs(img_original, arg):
    bn, c, h, w = img_original.shape
    # Make image and grid
    tps_param_dic = tps_parameters(bn, arg.scal, 0., 0., 0., 0., arg.augm_scal)
    coord, vector = make_input_tps_param(tps_param_dic)
    coord, vector = coord.to(arg.device), vector.to(arg.device)
    img, mesh = ThinPlateSpline(img_original, coord, vector, arg.reconstr_dim, device=arg.device)
    # Make transformed image and grid
    tps_param_dic_rot = tps_parameters(bn, arg.scal, arg.tps_scal, arg.rot_scal,
                                       arg.off_scal, arg.scal_var, arg.augm_scal)
    coord_rot, vector_rot = make_input_tps_param(tps_param_dic_rot)
    coord_rot, vector_rot = coord_rot.to(arg.device), vector_rot.to(arg.device)
    img_rot, mesh_rot = ThinPlateSpline(img_original, coord_rot, vector_rot, arg.reconstr_dim, device=arg.device)
    # Make augmentation
    img_stack = torch.cat([img, img_rot], dim=0)
    img_stack_augm = augm(img_stack, arg, arg.device)
    img_augm, img_rot_augm = img_stack_augm[:bn], img_stack_augm[bn:]

    # Make input stack
    input_images = F.interpolate(torch.cat([img_augm, img_rot], dim=0), size=arg.reconstr_dim).clamp(min=0., max=1.)
    reconstr_images = F.interpolate(torch.cat([img, img_rot_augm], dim=0), size=arg.reconstr_dim).clamp(min=0., max=1.)
    mesh_stack = torch.cat([mesh, mesh_rot], dim=0)

    return input_images, reconstr_images, mesh_stack


def get_local_part_appearances(f, sig):
    alpha = contract('bfij, bkij -> bkf', f, sig)
    return alpha


def rotation_mat(rotation):
    """
    :param rotation: tf tensor of shape [1]
    :return: rotation matrix as tf tensor with shape [2, 2]
    """
    a = torch.cos(rotation).unsqueeze(0)
    b = torch.sin(rotation).unsqueeze(0)
    row_1 = torch.cat((a, -b), 1)
    row_2 = torch.cat((b, a), 1)
    mat = torch.cat((row_1, row_2), 0)
    return mat


def get_mu(part_maps, device):
    """
        Calculate mean for each channel of part_maps
        :param part_maps: tensor of part map activations [bn, n_part, h, w]
        :return: mean calculated on a grid of scale [-1, 1]
        """
    bn, nk, h, w = part_maps.shape
    y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1).unsqueeze(-1)
    meshgrid = torch.cat((y_t, x_t), dim=-1) # 64 x 64 x 2

    mu = torch.einsum('akij, ijl -> akl', part_maps, meshgrid) # bn x nk x 2

    return mu


def get_mu_and_prec(part_maps, device, L_inv_scal):
    """
        Calculate mean for each channel of part_maps
        :param part_maps: tensor of part map activations [bn, n_part, h, w]
        :return: mean calculated on a grid of scale [-1, 1]
        """
    bn, nk, h, w = part_maps.shape
    y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1).unsqueeze(-1)
    meshgrid = torch.cat((y_t, x_t), dim=-1) # 64 x 64 x 2

    mu = contract('akij, ijl -> akl', part_maps, meshgrid) # bn x nk x 2
    mu_out_prod = contract('akm, akn -> akmn', mu, mu)

    mesh_out_prod = contract('ijm, ijn -> ijmn', meshgrid, meshgrid)
    stddev = contract('ijmn, akij -> akmn', mesh_out_prod, part_maps) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-12

    a = torch.sqrt(torch.abs(a_sq + eps))  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = torch.sqrt(torch.abs(b_sq_add_c_sq - b ** 2 + eps))
    z = torch.zeros_like(a)

    det = (a * c).unsqueeze(-1).unsqueeze(-1)
    row_1 = torch.cat((c.unsqueeze(-1), z.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    row_2 = torch.cat((-b.unsqueeze(-1), a.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    L_inv = L_inv_scal / (det + eps) * torch.cat((row_1, row_2), dim=-2)  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    L_inv = torch.clamp(L_inv, min=-1000., max=1000.)

    return mu, L_inv


def softmax(logit_map):
    bn, kn, h, w = logit_map.shape
    map_norm = nn.Softmax(dim=2)(logit_map.reshape(bn, kn, -1)).reshape(bn, kn, h, w)
    return map_norm


def get_heat_map(mu, L_inv, device, h=64):
    h, w, bn, nk = h, h, L_inv.shape[0], L_inv.shape[1]

    y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w)
    x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1)
    x_t_flat = x_t.reshape(1, 1, -1)
    y_t_flat = y_t.reshape(1, 1, -1)

    mesh = torch.cat([y_t_flat, x_t_flat], dim=-2)
    eps = 1e-6
    dist = mesh - mu.unsqueeze(-1) + eps

    proj_precision = contract('bnik, bnkf -> bnif', L_inv, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis
    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(bn, nk, h, w)  # bn number parts width height

    return heat


def precision_dist_op(precision, dist, part_depth, nk, h, w):
    proj_precision = contract('bnik, bnkf -> bnif', precision, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis
    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(-1, nk, h, w)  # bn number parts width height

    part_heat = heat[:, :part_depth]

    return heat, part_heat


def feat_mu_to_enc(features, mu, L_inv, device, reconstr_dim):
    bn, nk, nf = features.shape
    if reconstr_dim == 128:
        reconstruct_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feat_map_depths = [[0, 0], [0, 0], [0, 0], [4, nk], [2, 4], [0, 2]]
        part_depths = [nk, nk, nk, nk, 4, 2]
    elif reconstr_dim == 256:
        reconstruct_stages = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feat_map_depths = [[0, 0], [0, 0], [0, 0], [0, 0], [4, nf], [2, 4], [0, 2]]
        part_depths = [nk, nk, nk, nk, nk, 4, 2]

    encoding_list = []

    for dims, part_depth, feat_slice in zip(reconstruct_stages, part_depths, feat_map_depths):
        h, w = dims[0], dims[1]

        y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w).unsqueeze(-1)
        x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1).unsqueeze(-1)

        y_t_flat = y_t.reshape(1, 1, 1, -1)
        x_t_flat = x_t.reshape(1, 1, 1, -1)

        mesh = torch.cat((y_t_flat, x_t_flat), dim=-2)
        eps = 1e-6
        dist = mesh - mu.unsqueeze(-1) + eps

        heat_shape, part_heat_shape = precision_dist_op(L_inv, dist, part_depth, nk, h, w)

        nkf = feat_slice[1] - feat_slice[0]

        if nkf != 0:
            feature_slice_rev = features[:, feat_slice[0]: feat_slice[1]]
            heat_scal = heat_shape[:, feat_slice[0]: feat_slice[1]]
            heat_scal_norm = torch.sum(heat_scal, 1, keepdim=True) + 1
            heat_scal = heat_scal / heat_scal_norm
            heat_feat_map = contract('bkij,bkn -> bnij', heat_scal, feature_slice_rev)

            encoding_list.append(torch.cat((part_heat_shape, heat_feat_map), 1))

        else:
            encoding_list.append(part_heat_shape)

    return encoding_list


def heat_map_function(y_dist, x_dist, y_scale, x_scale):
    x = 1 / (1 + (torch.square(y_dist / (1e-6 + y_scale)) + torch.square(x_dist / (1e-6 + x_scale))))
    return x


def fold_img_with_L_inv(img, mu, L_inv, scale, threshold, device, normalize=True):
    bn, nc, h, w = img.shape
    bn, nk, _ = mu.shape

    mu_stop = mu.detach()

    y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w)
    x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1)
    x_t_flat = x_t.reshape(1, 1, -1)
    y_t_flat = y_t.reshape(1, 1, -1)

    mesh = torch.cat([y_t_flat, x_t_flat], dim=-2)
    eps = 1e-6
    dist = mesh - mu_stop.unsqueeze(-1) + eps

    proj_precision = contract('bnik, bnkf -> bnif', scale * L_inv, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis

    heat = 1 / (1 + proj_precision)

    heat = torch.reshape(heat, shape=[bn, nk, h, w])  # bn width height number parts
    heat = contract('bkij -> bij', heat)
    heat_scal = torch.clamp(heat, min=0., max=1.)
    heat_scal = torch.where(heat_scal > threshold, heat_scal, torch.zeros_like(heat_scal))

    norm = torch.sum(heat_scal.reshape(bn, -1), dim=1).unsqueeze(1).unsqueeze(1)
    if normalize:
        heat_scal = heat_scal / norm
    folded_img = contract('bcij, bij -> bcij', img, heat_scal)

    return folded_img


def loss_fn(bn, mu, L_inv, mu_t, stddev_t, reconstruct_same_id, image_rec,
            l_2_scal, l_2_threshold, L_mu, L_cov, L_rec, device):

    # Equiv Loss
    mu_t_1, mu_t_2 = mu_t[:bn], mu_t[bn:]
    bn, nk, _ = mu_t_1.shape
    stddev_t_1, stddev_t_2 = stddev_t[:bn], stddev_t[bn:]
    transform_loss = torch.mean((mu_t_1 - mu_t_2) ** 2)

    eps = 1e-7
    precision_sq = (stddev_t_1 - stddev_t_2) ** 2
    precision_loss = torch.mean(torch.sqrt(torch.sum(precision_sq, dim=[2, 3]) + eps))

    # Reconstruction Loss
    img_difference = reconstruct_same_id - image_rec
    distance_metric = torch.abs(img_difference)
    # Fold Image
    fold_img_squared = fold_img_with_L_inv(distance_metric, mu.detach(), L_inv.detach(),
                                           l_2_scal, l_2_threshold, device)
    rec_loss = torch.mean(torch.sum(fold_img_squared, dim=[2, 3]))

    # Get Total Loss
    total_loss = L_rec * rec_loss + L_mu * transform_loss + L_cov * precision_loss

    return total_loss, rec_loss, transform_loss, precision_loss


def coordinate_transformation(coords, grid, device, grid_size=1000):
    bn, k, _ = coords.shape
    bucket = torch.linspace(-10., 10., grid_size, device=device)
    indices = torch.bucketize(coords.contiguous(), bucket)
    indices = indices.unsqueeze(-2).unsqueeze(-2)
    grid = grid.unsqueeze(1).repeat(1, k, 1, 1, 1)
    new_coords = torch.gather(grid, 3, indices).squeeze(-2).squeeze(-2)

    return new_coords


if __name__ == "__main__":
    pass