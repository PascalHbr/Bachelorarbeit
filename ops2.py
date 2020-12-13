import torch
import torch.nn.functional as F
import kornia.augmentation as K
from opt_einsum import contract


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
    random_tensor = 1. - arg.p_flip + torch.rand(size=[1], dtype=t.dtype, device=device)
    binary_tensor = torch.floor(random_tensor)
    random_tensor, binary_tensor = random_tensor, binary_tensor

    augmented = binary_tensor * t + (1 - binary_tensor) * (1 - t)
    return augmented


def prepare_pairs(t_images, arg, device):
    if arg.mode == 'train':
        bn, n_c, w, h = t_images.shape
        t_c_1_images = augm(t_images, arg, device)
        t_c_2_images = augm(t_images, arg, device)

        if arg.static:
            t_c_1_images = torch.cat([t_c_1_images[:bn // 2].unsqueeze(1), t_c_1_images[bn // 2:].unsqueeze(1)], dim=1)
            t_c_2_images = torch.cat([t_c_2_images[:bn // 2].unsqueeze(1), t_c_2_images[bn // 2:].unsqueeze(1)], dim=1)
        else:
            t_c_1_images = t_c_1_images.reshape(bn // 2, 2, n_c, h, w)
            t_c_2_images = t_c_2_images.reshape(bn // 2, 2, n_c, h, w)

        a, b = t_c_1_images[:, 0].unsqueeze(1), t_c_1_images[:, 1].unsqueeze(1)
        c, d = t_c_2_images[:, 0].unsqueeze(1), t_c_2_images[:, 1].unsqueeze(1)

        if arg.static:
            t_input_images = torch.cat([a, d], dim=0).reshape(bn, n_c, w, h)
            t_reconstr_images = torch.cat([c, b], dim=0).reshape(bn, n_c, w, h)
        else:
            t_input_images = torch.cat([a, d], dim=1).reshape(bn, n_c, w, h)
            t_reconstr_images = torch.cat([c, b], dim=1).reshape(bn, n_c, w, h)

        t_input_images = torch.clamp(t_input_images, min=0., max=1.)
        t_reconstr_images = F.interpolate(torch.clamp(t_reconstr_images, min=0., max=1.), size=arg.reconstr_dim)

    else:
        t_input_images = torch.clamp(t_images, min=0., max=1.)
        t_reconstr_images = F.interpolate(torch.clamp(t_images, min=0., max=1.), size=arg.reconstr_dim)

    return t_input_images, t_reconstr_images


def heat_map_function(y_dist, x_dist, y_scale, x_scale):
    x = 1 / (1 + (torch.square(y_dist / (1e-6 + y_scale)) + torch.square(x_dist / (1e-6 + x_scale))))
    return x


def fold_img_with_mu(img, mu, scale, threshold, device, normalize=True):
    """
        folds the pixel values of img with potentials centered around the part means (mu)
        :param img: batch of images
        :param mu:  batch of part means in range [-1, 1]
        :param scale: scale that governs the range of the potential
        :param normalize: whether to normalize the potentials
        :return: folded image
        """
    bn, nc, h, w = img.shape
    bn, nk, _ = mu.shape

    py = mu[:, :, 0].unsqueeze(2)
    px = mu[:, :, 1].unsqueeze(2)

    y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w)
    x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1)
    x_t_flat = x_t.reshape(1, 1, -1)
    y_t_flat = y_t.reshape(1, 1, -1)

    eps = 1e-6
    y_dist = py - y_t_flat + eps
    x_dist = px - x_t_flat + eps

    heat_scal = heat_map_function(y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale)
    heat_scal = heat_scal.reshape(bn, nk, h, w)  # bn width height number parts
    heat_scal = contract("bkij -> bij", heat_scal)
    heat_scal = torch.clamp(heat_scal, min=0., max=1.)
    heat_scal = torch.where(heat_scal > threshold, heat_scal, torch.zeros_like(heat_scal))

    norm = torch.sum(heat_scal.reshape(bn, -1), dim=1).unsqueeze(1).unsqueeze(1)
    if normalize:
        heat_scal_norm = heat_scal / norm
        folded_img = contract('bcij,bij->bcij', img, heat_scal_norm)
    if not normalize:
        folded_img = contract('bcij,bij->bcij', img, heat_scal)

    return folded_img, heat_scal.unsqueeze(-1)


def fold_img_with_L_inv(img, mu, L_inv, scale, threshold, device, normalize=True):
    """
        folds the pixel values of img with potentials centered around the part means (mu)
        :param img: batch of images
        :param mu:  batch of part means in range [-1, 1]
        :param scale: scale that governs the range of the potential
        :param normalize: whether to normalize the potentials
        :return: folded image
        """
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


def loss_fn(bn, mu, L_inv, mu_t, stddev_t, reconstruct_same_id, image_rec, fold_with_shape, l_2_scal, l_2_threshold,
            L_mu, L_cov, device):

    # Equiv Loss
    mu_t_1, mu_t_2 = mu_t[:bn], mu_t[bn:]
    stddev_t_1, stddev_t_2 = stddev_t[:bn], stddev_t[bn:]
    transform_loss = torch.mean((mu_t_1 - mu_t_2) ** 2)

    precision_sq = (stddev_t_1 - stddev_t_2) ** 2

    eps = 1e-6
    precision_loss = torch.mean(torch.sqrt(torch.sum(precision_sq, dim=[2, 3]) + eps))

    img_difference = reconstruct_same_id - image_rec
    distance_metric = torch.abs(img_difference)

    if fold_with_shape:
        fold_img_squared = fold_img_with_L_inv(distance_metric, mu.detach(), L_inv.detach(),
                                               l_2_scal, l_2_threshold, device)
    else:
        fold_img_squared, heat_mask_l2 = fold_img_with_mu(distance_metric, mu, l_2_scal, l_2_threshold, device)

    rec_loss = torch.mean(torch.sum(fold_img_squared, dim=[2, 3]))

    total_loss = rec_loss + L_mu * transform_loss + L_cov * precision_loss
    return total_loss, rec_loss, transform_loss, precision_loss