import torch.nn.functional as F
import torch
from dotmap import DotMap
from opt_einsum import contract


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


def tps_parameters(batch_size, scal, tps_scal, rot_scal, off_scal, scal_var, augm_scal, rescal=1.):
    coord = torch.tensor([[[-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5], [0.5, 0.5],
                         [0.2, -0.2], [-0.2, 0.2], [0.2, 0.2], [-0.2, - 0.2]]],
                         dtype=torch.float32)
    coord = coord.repeat(batch_size, 1, 1)
    shape = coord.shape
    coord = coord + (-0.2 - 0.2) * torch.rand(size=shape) + 0.2
    vector = (-tps_scal - tps_scal) * torch.rand(size=shape, dtype=torch.float32) + tps_scal

    offset = (-off_scal - off_scal) * torch.rand(size=[batch_size, 1, 2], dtype=torch.float32) + off_scal
    offset_2 = (-off_scal - off_scal) * torch.rand(size=[batch_size, 1, 2], dtype=torch.float32) + off_scal
    t_scal = (scal * (1. - scal_var) - scal * (1. + scal_var)) * torch.rand(size=[batch_size, 2], dtype=torch.float32) \
             + scal * (1. + scal_var)
    t_scal = t_scal * rescal

    rot_param = (-rot_scal - rot_scal) * torch.rand(size=[batch_size, 1], dtype=torch.float32) + rot_scal
    rot_mat = torch.cat([rotation_mat(rot_param[i]).unsqueeze(0) for i in range(rot_param.shape[0])], 0)

    parameter_dict = {'coord': coord, 'vector': vector, 'offset': offset, 'offset_2': offset_2,
                      't_scal': t_scal, 'rot_mat': rot_mat, 'augm_scal': augm_scal}
    parameter_dict = DotMap(parameter_dict)
    return parameter_dict


def make_input_tps_param(tps_param, move_point=None, scal_point=None):
    '''

    '''
    coord = tps_param.coord
    vector = tps_param.vector
    offset = tps_param.offset
    offset_2 = tps_param.offset_2
    rot_mat = tps_param.rot_mat
    t_scal = tps_param.t_scal

    scaled_coord = contract('bk,bck->bck', t_scal, coord + vector - offset) + offset
    t_vector = contract('blk,bck->bcl', rot_mat, scaled_coord - offset_2) + offset_2 - coord

    if move_point is not None and scal_point is not None:
        coord = contract('bk,bck->bck', scal_point, coord + move_point)
        t_vector = contract('bk,bck->bck', scal_point, t_vector)

    else:
        assert(move_point is None and scal_point is None)

    return coord, t_vector


def ThinPlateSpline(U, coord, vector, out_size, move=None, scal=None):
    device = U.get_device()
    coord = torch.flip(coord, [2])
    vector = torch.flip(vector, [2])

    num_batch, channels, height, width = U.shape
    out_height = out_size
    out_width = out_size
    height_f = torch.tensor([height], dtype=torch.float32, device=device)
    width_f = torch.tensor([width], dtype=torch.float32, device=device)
    num_point = coord.shape[1]

    def _repeat(x, n_repeats):
        x = x.to(dtype=torch.float32)
        rep = torch.ones(n_repeats, dtype=torch.float32, device=device).unsqueeze(0)
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        return torch.reshape(x, [-1])

    def _interpolate(im, y, x):
        y = y.to(dtype=torch.float32, device=device)
        x = x.to(dtype=torch.float32, device=device)

        zero = 0
        max_y = height - 1
        max_x = width - 1

        # scale indices from aprox [-1, 1] to [0, width/height]
        y = (y + 1) * height_f / 2.0
        x = (x + 1) * width_f / 2.0

        y = torch.reshape(y, [-1])
        x = torch.reshape(x, [-1])

        y0 = torch.floor(y).to(dtype=torch.int32)
        y1 = y0 + 1
        x0 = torch.floor(x).to(dtype=torch.int32)
        x1 = x0 + 1

        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)

        base = _repeat(torch.arange(num_batch, device=device) * width * height, out_height * out_width)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        idx_a = (base_y0 + x0).to(dtype=torch.int64)
        idx_b = (base_y1 + x0).to(dtype=torch.int64)
        idx_c = (base_y0 + x1).to(dtype=torch.int64)
        idx_d = (base_y1 + x1).to(dtype=torch.int64)

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im = im.permute(0, 2, 3, 1).contiguous()
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.to(dtype=torch.float32)

        Ia = torch.index_select(im_flat, 0, idx_a)
        Ib = torch.index_select(im_flat, 0, idx_b)
        Ic = torch.index_select(im_flat, 0, idx_c)
        Id = torch.index_select(im_flat, 0, idx_d)

        # and finally calculate interpolated values
        x0_f = x0.to(dtype=torch.float32, device=device)
        x1_f = x1.to(dtype=torch.float32, device=device)
        y0_f = y0.to(dtype=torch.float32, device=device)
        y1_f = y1.to(dtype=torch.float32, device=device)

        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        output = wa * Ia + wb * Ib + wc * Ic + wd * Id
        return output

    def _meshgrid(height, width, coord):

        x_t = torch.reshape(torch.linspace(- 1., 1., width, device=device), [1, width]).repeat(height, 1)
        y_t = torch.reshape(torch.linspace(- 1., 1., height, device=device), [height, 1]).repeat(1, width)

        x_t_flat = torch.reshape(x_t, (1, 1, -1))
        y_t_flat = torch.reshape(y_t, (1, 1, -1))

        px = coord[:, :, 0].unsqueeze(2)  # [bn, pn, 1]
        py = coord[:, :, 1].unsqueeze(2)  # [bn, pn, 1]
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6)  # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.repeat(num_batch, 1, 1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.repeat(num_batch, 1, 1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g)  # [bn, 1, h*w]

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1)  # [bn, 3+pn, h*w]
        return grid

    def _transform(T, coord, move, scal):
        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        grid = _meshgrid(out_height, out_width, coord)  # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)  #
        x_s = T_g[:, :1, :]
        y_s = T_g[:, 1:2, :]

        if move is not None and scal is not None:
            off_y = move[:, :, 0].unsqueeze(1)
            off_x = move[:, :, 1].unsqueeze(1)
            scal_y = scal[:, 0].unsqueeze(-1).unsqueeze(-1)
            scal_x = scal[:, 1].unsqueeze(-1).unsqueeze(-1)
            y = (y_s * scal_y + off_y)
            x = (x_s * scal_x + off_x)

        else:
            assert (move is None and scal is None)
            y = y_s
            x = x_s

        return y, x

    def _solve_system(coord, vector):
        ones = torch.ones((num_batch, num_point, 1), dtype=torch.float32, device=device)
        p = torch.cat((ones, coord), 2)  # [bn, pn, 3]

        p_1 = torch.reshape(p, [num_batch, -1, 1, 3])  # [bn, pn, 1, 3]
        p_2 = torch.reshape(p, [num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1 - p_2), 3)  # [bn, pn, pn]
        r = d2 * torch.log(d2 + 1e-6)  # Kernel [bn, pn, pn]

        zeros = torch.zeros((num_batch, 3, 3), dtype=torch.float32, device=device)
        W_0 = torch.cat((p, r), 2)  # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0, 2, 1)), 2)  # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1)  # [bn, pn+3, pn+3]
        W_inv = torch.inverse(W)

        tp = F.pad(coord + vector, [0, 0, 0, 3, 0, 0], "constant", 0)  # [bn, pn+3, 2]
        T = torch.matmul(W_inv, tp)  # [bn, pn+3, 2]
        T = T.permute(0, 2, 1)  # [bn, 2, pn+3]

        return T

    T = _solve_system(coord, vector)
    y, x = _transform(T, coord, move, scal)
    input_transformed = _interpolate(U, y, x)
    output = torch.reshape(input_transformed, [num_batch, out_height, out_width, channels]).permute(0, 3, 1, 2).contiguous()
    output = torch.clamp(output, min=0., max=1.)
    y = torch.reshape(y, [num_batch, 1, out_height, out_width])
    x = torch.reshape(x, [num_batch, 1, out_height, out_width])
    t_arr = torch.cat([y, x], 1)
    return output, t_arr
