import torch
import torch.nn as nn
from transformations import ThinPlateSpline
import kornia.augmentation as K
import torch.nn.functional as F
from opt_einsum import contract
from architecture_ops import softmax


def get_local_part_appearances(f, sig):
    alpha = contract('bfij, bkij -> bkf', f, sig)
    return alpha


def get_covariance(tensor):
    bn, nk, w, h = tensor.shape
    tensor_reshape = tensor.reshape(bn, nk, 2, -1)
    x = tensor_reshape[:, :, 0, :]
    y = tensor_reshape[:, :, 1, :]
    mean_x = torch.mean(x, dim=2).unsqueeze(-1)
    mean_y = torch.mean(y, dim=2).unsqueeze(-1)

    xx = torch.sum((x - mean_x) * (x - mean_x), dim=2).unsqueeze(-1) / (h * w / 2 - 1)
    xy = torch.sum((x - mean_x) * (y - mean_y), dim=2).unsqueeze(-1) / (h * w / 2 - 1)
    yx = xy
    yy = torch.sum((y - mean_y) * (y - mean_y), dim=2).unsqueeze(-1) / (h * w / 2 - 1)

    cov = torch.cat((xx, xy, yx, yy), dim=2)
    cov = cov.reshape(bn, nk, 2, 2)
    return cov


def get_mu_and_prec(part_maps, device, scal):
    """
        Calculate mean for each channel of part_maps
        :param part_maps: tensor of part map activations [bn, n_part, h, w]
        :return: mean calculated on a grid of scale [-1, 1]
        """
    bn, nk, h, w = part_maps.shape
    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)
    meshgrid = torch.cat((y_t, x_t), dim=-1).to(device) # 64 x 64 x 2

    mu = contract('ijl, akij -> akl', meshgrid, part_maps) # bn x nk x 2
    mu_out_prod = contract('akm,akn->akmn', mu, mu)

    mesh_out_prod = contract('ijm,ijn->ijmn', meshgrid, meshgrid)
    stddev = contract('ijmn,akij->akmn', mesh_out_prod, part_maps) - mu_out_prod

    a_sq = stddev[:, :, 0, 0]
    a_b = stddev[:, :, 0, 1]
    b_sq_add_c_sq = stddev[:, :, 1, 1]
    eps = 1e-6

    a = torch.sqrt(torch.abs(a_sq) + eps)  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = torch.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = torch.zeros_like(a)
    det = (a * c).unsqueeze(-1).unsqueeze(-1)
    row_1 = torch.cat((c.unsqueeze(-1), z.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    row_2 = torch.cat((-b.unsqueeze(-1), a.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    L_inv = scal / (det + eps) * torch.cat((row_1, row_2), dim=-2)  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    return mu, L_inv


def get_heat_map(mu, L_inv, device):
    h, w, nk = 64, 64, L_inv.shape[1]

    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)

    y_t_flat = y_t.reshape(1, 1, 1, -1)
    x_t_flat = x_t.reshape(1, 1, 1, -1)

    mesh = torch.cat((y_t_flat, x_t_flat), dim=-2).to(device)
    dist = mesh - mu.unsqueeze(-1)
    proj_precision = contract('bnik, bnkf -> bnif', L_inv, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis
    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(-1, nk, h, w)  # bn number parts width height

    return heat


def precision_dist_op(precision, dist, part_depth, nk, h, w):
    proj_precision = contract('bnik, bnkf -> bnif', precision, dist) ** 2  # tf.matmul(precision, dist)**2
    proj_precision = torch.sum(proj_precision, -2)  # sum x and y axis
    heat = 1 / (1 + proj_precision)
    heat = heat.reshape(-1, nk, h, w)  # bn number parts width height
    part_heat = heat[:, :part_depth]
    return heat, part_heat


def reverse_batch(tensor, n_reverse):
    """
    reverses order of elements the first axis of tensor
    example: reverse_batch(tensor=tf([[1],[2],[3],[4],[5],[6]), n_reverse=3) returns tf([[3],[2],[1],[6],[5],[4]]) for n reverse 3
    :param tensor:
    :param n_reverse:
    :return:
    """
    bn, rest = tensor.shape[0], tensor.shape[1:]
    assert ((bn / n_reverse).is_integer())
    tensor = torch.reshape(tensor, shape=[bn // n_reverse, n_reverse, *rest])
    tensor_rev = tensor.flip(dims=[1])
    tensor_rev = torch.reshape(tensor_rev, shape=[bn, *rest])
    return tensor_rev


def feat_mu_to_enc(features, mu, L_inv, device, covariance, reconstr_dim, static=True, n_reverse=2, feat_shape=True,
                   heat_feat_normalize=True, range=10):
    """
    :param features: tensor shape   bn, nk, nf
    :param mu: tensor shape  [bn, nk, 2] in range[-1,1]
    :param L_inv: tensor shape  [bn, nk, 2, 2]
    :param n_reverse:
    :return:
    """
    bn, nk, nf = features.shape
    if reconstr_dim == 128:
        reconstruct_stages = [[128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feat_map_depths = [[0, 0], [0, 0], [0, 0], [4, nk], [2, 4], [0, 2]]
        part_depths = [nk, nk, nk, nk, 4, 2]
    elif reconstr_dim == 256:
        reconstruct_stages = [[256, 256], [128, 128], [64, 64], [32, 32], [16, 16], [8, 8], [4, 4]]
        feat_map_depths = [[0, 0], [0, 0], [0, 0], [0, 0], [4, nf], [2, 4], [0, 2]]
        part_depths = [nk, nk, nk, nk, nk, 4, 2]

    if static:
        #reverse_features = torch.cat([features[bn // 2:], features[:bn // 2]], dim=0)
        reverse_features = features
    else:
        reverse_features = reverse_batch(features, n_reverse)

    encoding_list = []
    circular_precision = range * torch.eye(2).reshape(1, 1, 2, 2).to(dtype=torch.float).repeat(bn, nk, 1, 1).to(device)


    for dims, part_depth, feat_slice in zip(reconstruct_stages, part_depths, feat_map_depths):
        h, w = dims[0], dims[1]

        y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w).unsqueeze(-1)
        x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1).unsqueeze(-1)

        y_t_flat = y_t.reshape(1, 1, 1, -1)
        x_t_flat = x_t.reshape(1, 1, 1, -1)

        mesh = torch.cat((y_t_flat, x_t_flat), dim=-2).to(device)
        dist = mesh - mu.unsqueeze(-1)

        if not covariance or not feat_shape:
            heat_circ, part_heat_circ = precision_dist_op(circular_precision, dist, part_depth, nk, h, w)

        if covariance or feat_shape:
            heat_shape, part_heat_shape = precision_dist_op(L_inv, dist, part_depth, nk, h, w)

        nkf = feat_slice[1] - feat_slice[0]

        if nkf != 0:
            feature_slice_rev = reverse_features[:, feat_slice[0]: feat_slice[1]]

            if feat_shape:
                heat_scal = heat_shape[:, feat_slice[0]: feat_slice[1]]

            else:
                heat_scal = heat_circ[:, feat_slice[0]: feat_slice[1]]

            if heat_feat_normalize:
                heat_scal_norm = torch.sum(heat_scal, 1, keepdim=True) + 1
                heat_scal = heat_scal / heat_scal_norm

            heat_feat_map = contract('bkij,bkn->bnij', heat_scal, feature_slice_rev)


            if covariance:
                encoding_list.append(torch.cat((part_heat_shape, heat_feat_map), 1))

            else:
                encoding_list.append(torch.cat((part_heat_circ, heat_feat_map), 1))

        else:
            if covariance:
                encoding_list.append(part_heat_shape)

            else:
                encoding_list.append(part_heat_circ)

    return encoding_list


def total_loss(input, reconstr, sig_shape_raw, sig_app, mu, coord, vector,
               device, L_mu, L_cov, scal, l_2_scal, l_2_threshold):
    bn, k, h, w = sig_shape_raw.shape
    # Equiv Loss
    sig_shape_trans, _ = ThinPlateSpline(sig_shape_raw, coord, vector, h, device=device)
    sig_shape = softmax(sig_shape_trans)
    mu_1, L_inv1 = get_mu_and_prec(sig_app, device, scal)
    #cov_1 = get_covariance(sig_app)
    mu_2, L_inv2 = get_mu_and_prec(sig_shape, device, scal)
    #cov_2 = get_covariance(sig_shape_trans)
    equiv_loss = torch.mean(torch.sum(L_mu * torch.norm(mu_1 - mu_2, p=2, dim=2) + \
                           L_cov * torch.norm(L_inv1 - L_inv2, p=1, dim=[2, 3]), dim=1))

    # Rec Loss
    distance_metric = torch.abs(input - reconstr)
    fold_img_squared, heat_mask_l2 = fold_img_with_mu(distance_metric, mu, l_2_scal, l_2_threshold, device)
    rec_loss = torch.mean(torch.sum(torch.sum(fold_img_squared.reshape(bn, k, -1), dim=2), dim=1))
    total_loss = rec_loss + equiv_loss
    return total_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augm(t, arg):
    t = K.ColorJitter(arg.brightness_var, arg.contrast_var, arg.saturation_var, arg.hue_var)(t)
    random_tensor = 1. - arg.p_flip + torch.rand(size=[1], dtype=t.dtype)
    binary_tensor = torch.floor(random_tensor)
    random_tensor, binary_tensor = random_tensor.to(arg.device), binary_tensor.to(arg.device)

    augmented = binary_tensor * t + (1 - binary_tensor) * (1 - t)
    return augmented


def prepare_pairs(t_images, arg, reconstr_dim=128):
    if arg.mode == 'train':
        bn, n_c, w, h = t_images.shape
        t_c_1_images = augm(t_images, arg)
        t_c_2_images = augm(t_images, arg)

        if arg.static:
            t_c_1_images = torch.cat([t_c_1_images[:bn//2].unsqueeze(1), t_c_1_images[bn//2:].unsqueeze(1)], dim=1)
            t_c_2_images = torch.cat([t_c_2_images[:bn//2].unsqueeze(1), t_c_2_images[bn//2:].unsqueeze(1)], dim=1)
        else:
            t_c_1_images = t_c_1_images.reshape(bn // 2, 2, n_c, h, w)
            t_c_2_images = t_c_2_images.reshape(bn // 2, 2, n_c, h, w)

        a, b = t_c_1_images[:, 0].unsqueeze(1), t_c_1_images[:, 1].unsqueeze(1)
        c, d = t_c_2_images[:, 0].unsqueeze(1), t_c_2_images[:, 1].unsqueeze(1)

        if arg.static:
            t_input_images = torch.cat([a, d], dim=0).reshape(bn, n_c, w, h)
            t_reconst_images = torch.cat([c, b], dim=0).reshape(bn, n_c, w, h)
        else:
            t_input_images = torch.cat([a, d], dim=1).reshape(bn, n_c, w, h)
            t_reconst_images = torch.cat([c, b], dim=1).reshape(bn, n_c, w, h)

        t_input_images = torch.clamp(t_input_images, min=0., max=1.)
        t_reconst_images = F.interpolate(torch.clamp(t_reconst_images, min=0., max=1.), size=reconstr_dim)

    else:
        t_input_images = torch.clamp(t_images, min=0., max=1.)
        t_reconst_images = F.interpolate(torch.clamp(t_images, min=0., max=1.), size=reconstr_dim)

    return t_input_images, t_reconst_images


def AbsDetJacobian(batch_meshgrid, device):
    """
        :param batch_meshgrid: takes meshgrid tensor of dim [bn, 2, h, w] (conceptually meshgrid represents a two dimensional function f = [fx, fy] on [bn, h, w] )
        :return: returns Abs det of  Jacobian of f of dim [bn, 1, h, w]
        """
    y_c = batch_meshgrid[:, 0, :, :].unsqueeze(1)
    x_c = batch_meshgrid[:, 1, :, :].unsqueeze(1)
    sobel_x_filter = 1 / 4 * torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float).reshape(1, 1, 3, 3).cuda()
    sobel_y_filter = sobel_x_filter.permute(0, 1, 3, 2).cuda()


    filtered_y_y = F.conv2d(y_c, sobel_y_filter, stride=1, padding=1)
    filtered_y_x = F.conv2d(y_c, sobel_x_filter, stride=1, padding=1)
    filtered_x_y = F.conv2d(x_c, sobel_y_filter, stride=1, padding=1)
    filtered_x_x = F.conv2d(x_c, sobel_x_filter, stride=1, padding=1)

    Det = torch.abs(filtered_y_y * filtered_x_x - filtered_y_x * filtered_x_y)

    return Det


def heat_map_function(y_dist, x_dist, y_scale, x_scale):
    x = 1 / (1 + (torch.square((torch.abs(y_dist) + 1e-6) / y_scale) + torch.square(
        (torch.abs(x_dist) + 1e-6) / x_scale)))
    return x


def fold_img_with_mu(img, mu, scale, threshold, device, normalize=True):
    """
        folds the pixel values of img with potentials centered around the part means (mu)
        :param img: batch of images
        :param mu:  batch of part means in range [-1, 1]
        :param scale: scale that governs the range of the potential
        :param visualize:
        :param normalize: whether to normalize the potentials
        :return: folded image
        """
    bn, nc, h, w = img.shape
    bn, nk, _ = mu.shape

    py = mu[:, :, 0].unsqueeze(2)
    px = mu[:, :, 1].unsqueeze(2)

    y_t = torch.linspace(-1., 1., h).reshape(h, 1).repeat(1, w)
    x_t = torch.linspace(-1., 1., w).reshape(1, w).repeat(h, 1)
    x_t_flat = x_t.reshape(1, 1, -1).to(device)
    y_t_flat = y_t.reshape(1, 1, -1).to(device)

    y_dist = py - y_t_flat
    x_dist = px - x_t_flat

    heat_scal = heat_map_function(y_dist=y_dist, x_dist=x_dist, x_scale=scale, y_scale=scale)
    heat_scal = heat_scal.reshape(bn, nk, h, w)  # bn width height number parts
    heat_scal = contract("bkij -> bij", heat_scal)
    heat_scal = torch.clamp(heat_scal, min=0., max=1.)
    heat_scal = torch.where(heat_scal > threshold, heat_scal, torch.zeros_like(heat_scal))

    eps = 1e-6
    norm = torch.sum(heat_scal.reshape(bn, -1), dim=1).unsqueeze(1).unsqueeze(1)
    if normalize:
        heat_scal_norm = heat_scal / (norm + eps)
        folded_img = contract('bcij,bij->bcij', img, heat_scal_norm)
    if not normalize:
        folded_img = contract('bcij,bij->bcij', img, heat_scal)

    return folded_img, heat_scal.unsqueeze(-1)


def normalize(image):
    bn, kn, h, w = image.shape
    image = image.view(bn, kn, -1)
    image -= image.min(2, keepdim=True)[0]
    image /= image.max(2, keepdim=True)[0]
    image = image.view(bn, kn, h, w)

    return image