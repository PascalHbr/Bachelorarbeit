import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from opt_einsum import contract
import os
from PIL import Image


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


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp


def plot_tensor(tensor):
    np_tensor = convert_image_np(tensor)
    plt.imshow(np_tensor)
    plt.show()


def batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])     # does that work?
    colour = torch.tensor(colour, dtype=torch.float).to(device)
    colour_map = contract('bkij, kl -> blij', heat_map, colour)
    return colour_map


def np_batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour).to(device)
    colour_map = contract('bkij,kl->blij', heat_map, np_colour)
    return colour_map


def identify_parts(image, raw, n_parts, version):
    image_base = np.array(Image.fromarray(image[0]).resize((64, 64))) / 255.
    base = image_base[:, :, 0] + image_base[:, :, 1] + image_base[:, :, 2]
    directory = os.path.join('../images/' + str(version) + "/identify/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(n_parts):
        plt.imshow(raw[0, :, :, i] + 0.02 * base, cmap='gray')
        fname = directory + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')


def save(img, mu, counter, model_save_dir):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join(model_save_dir + '/predictions/landmarks/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    n_parts = mu.shape[-2]
    mu_img = (mu + 1.) / 2. * np.array(out_shape)[0]
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
        plt.imshow(img[i])
        for j in range(n_parts):
            plt.scatter(mu_img[i, j, 1], mu_img[i, j, 0],  s=s, marker=marker_list[np.mod(j, len(marker_list))], color=cm.hsv(float(j / n_parts)))

        plt.axis('off')
        fname = directory + str(counter) + '_' + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()


def part_to_color_map(encoding_list, part_depths, size, device, square=True):
    part_maps = encoding_list[0][:, :part_depths[0], :, :]
    if square:
        part_maps = part_maps ** 4
    color_part_map = batch_colour_map(part_maps, device)
    color_part_map = torch.nn.Upsample(size=(size, size))(color_part_map)

    return color_part_map



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
        _, part_maps, sum_part_maps = self.E_sigma(image_in)
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


def main2(arg):
    # Get args
    bn = arg.bn
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
    weight_decay = arg.weight_decay
    epochs = arg.epochs
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    arg.device = device

    if mode == 'train':
        # Make new directory
        model_save_dir = name + "/training2"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/image')
            os.makedirs(model_save_dir + '/reconstruction')
            os.makedirs(model_save_dir + '/mu')
            os.makedirs(model_save_dir + '/parts')

        # Load Datasets
        train_data = load_images_from_folder()[:100]
        train_dataset = ImageDataset2(train_data, arg)
        test_data = load_images_from_folder()[-1000:]
        test_dataset = ImageDataset2(test_data, arg)

        # Prepare Dataloader & Instances
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=bn)
        model = Model2(arg).to(device)
        if load_from_ckpt == True:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Make Training
        for epoch in range(epochs):
            # Train on Train Set
            model.train()
            model.mode = 'train'
            for step, original in enumerate(train_loader):
                original = original.to(device, dtype=torch.float)
                optimizer.zero_grad()
                # plot_tensor(original[0])
                # plot_tensor(spat[0])
                # plot_tensor(app[0])
                # plot_tensor(original[1])
                # plot_tensor(spat[1])
                # plot_tensor(app[1])
                # print(coord, vec)
                prediction, loss = model(original)
                loss.backward()
                optimizer.step()
                if epoch % 2 == 0 and step == 0:
                    print(f'Epoch: {epoch}, Train Loss: {loss}')

            # Evaluate on Test Set
            model.eval()
            for step, original in enumerate(test_loader):
                with torch.no_grad():
                    original = original.to(device, dtype=torch.float)
                    prediction, loss = model(original)
                    if epoch % 2 == 0 and step == 0:
                        print(f'Epoch: {epoch}, Test Loss: {loss}')

            # Track Progress
            if epoch % 5 == 0:
                model.mode = 'predict'
                image, reconstruction, mu, shape_stream_parts, heat_map = model(original)
                for i in range(len(image)):
                    save_image(image[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
                    save_image(reconstruction[i], model_save_dir + '/image/' + str(i) + '_' + str(epoch) + '.png')
                    #save_image(mu[i], model_save_dir + '/image/' + str(epoch) + '.png')
                    #save_image(shape_stream_parts[i], model_save_dir + '/image/' + str(epoch) + '.png')

            # Save the current Model
            if epoch % 50 == 0:
                save_model(model, model_save_dir)

    elif arg.mode == 'predict':
        model_save_dir = arg.name + "/prediction"
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)


import torch
from transformations import ThinPlateSpline
from opt_einsum import contract
from architecture import softmax


def get_local_part_appearances(f, sig):
    alpha = contract('bfij, bkij -> bkf', f, sig)
    return alpha


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

    a = torch.sqrt(a_sq + eps)  # Σ = L L^T Prec = Σ^-1  = L^T^-1 * L^-1  ->looking for L^-1 but first L = [[a, 0], [b, c]
    b = a_b / (a + eps)
    c = torch.sqrt(b_sq_add_c_sq - b ** 2 + eps)
    z = torch.zeros_like(a)

    det = (a * c).unsqueeze(-1).unsqueeze(-1)
    row_1 = torch.cat((c.unsqueeze(-1), z.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    row_2 = torch.cat((-b.unsqueeze(-1), a.unsqueeze(-1)), dim=-1).unsqueeze(-2)
    L_inv = L_inv_scal / (det + eps) * torch.cat((row_1, row_2), dim=-2)  # L^⁻1 = 1/(ac)* [[c, 0], [-b, a]
    return mu, L_inv


def get_heat_map(mu, L_inv, device):
    h, w, bn, nk = 64, 64, L_inv.shape[0], L_inv.shape[1]

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
        # reverse_features = torch.cat([features[bn // 2:], features[:bn // 2]], dim=0)
        reverse_features = features
    else:
        # reverse_features = reverse_batch(features, n_reverse)
        reverse_features = torch.cat([features[bn // 2:], features[:bn // 2]], dim=0)

    encoding_list = []
    circular_precision = range * torch.eye(2, device=device).reshape(1, 1, 2, 2).to(dtype=torch.float).repeat(bn, nk, 1, 1)


    for dims, part_depth, feat_slice in zip(reconstruct_stages, part_depths, feat_map_depths):
        h, w = dims[0], dims[1]

        y_t = torch.linspace(-1., 1., h, device=device).reshape(h, 1).repeat(1, w).unsqueeze(-1)
        x_t = torch.linspace(-1., 1., w, device=device).reshape(1, w).repeat(h, 1).unsqueeze(-1)

        y_t_flat = y_t.reshape(1, 1, 1, -1)
        x_t_flat = x_t.reshape(1, 1, 1, -1)

        mesh = torch.cat((y_t_flat, x_t_flat), dim=-2)
        eps = 1e-6
        dist = mesh - mu.unsqueeze(-1) + eps

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

            heat_feat_map = contract('bkij,bkn -> bnij', heat_scal, feature_slice_rev)


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


def total_loss(input, reconstr, sig_shape_raw, sig_app, mu, L_inv, coord, vector,
               device, L_mu, L_cov, scal, l_2_scal, l_2_threshold, fold_with_shape):
    bn, k, h, w = sig_shape_raw.shape
    # Equiv Loss
    sig_shape_trans, _ = ThinPlateSpline(sig_shape_raw, coord, vector, h, device=device)
    sig_shape = softmax(sig_shape_trans)
    mu_1, L_inv1 = get_mu_and_prec(sig_app, device, scal)
    mu_2, L_inv2 = get_mu_and_prec(sig_shape, device, scal)
    equiv_loss = torch.mean(torch.sum(L_mu * torch.norm(mu_1 - mu_2, p=2, dim=2) + \
                            L_cov * torch.norm(L_inv1 - L_inv2, p=1, dim=[2, 3]), dim=1))

    # Rec Loss
    distance_metric = torch.abs(input - reconstr)
    if fold_with_shape:
        fold_img_squared = fold_img_with_L_inv(distance_metric, mu.detach(), L_inv.detach(),
                                               l_2_scal, l_2_threshold, device)
    else:
        fold_img_squared, heat_mask_l2 = fold_img_with_mu(distance_metric, mu, l_2_scal, l_2_threshold, device)

    rec_loss = torch.mean(torch.sum(fold_img_squared, dim=[2, 3]))
    total_loss = rec_loss + equiv_loss
    return total_loss, rec_loss, equiv_loss


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


def normalize(image):
    bn, kn, h, w = image.shape
    image = image.reshape(bn, kn, -1)
    image -= image.min(2, keepdim=True)[0]
    image /= image.max(2, keepdim=True)[0]
    image = image.reshape(bn, kn, h, w)
    return image


import torch.nn as nn
from architecture import E, Decoder
from ops_old import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec, total_loss


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


import torch
from Dataloader import ImageDataset, DataLoader
from utils import save_model, load_model, load_deep_fashion_dataset, make_visualization
from Model_old import Model
from config import parse_args, write_hyperparameters
from dotmap import DotMap
from ops_old import normalize
import os
import numpy as np
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
import kornia.augmentation as K
import wandb


def main(arg):
    # Set random seeds
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)

    # Get args
    bn = arg.bn
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
    epochs = arg.epochs
    device = torch.device('cuda:' + str(arg.gpu) if torch.cuda.is_available() else 'cpu')
    arg.device = device

    if mode == 'train':
        # Make new directory
        model_save_dir = '../results/' + name
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/summary')

        # Save Hyperparameters
        write_hyperparameters(arg.toDict(), model_save_dir)

        # Define Model & Optimizer
        model = Model(arg).to(device)
        if load_from_ckpt:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Log with wandb
        wandb.init(project='Disentanglement', config=arg, name=arg.name)
        wandb.watch(model, log='all')
        # Load Datasets and DataLoader
        train_data, test_data = load_deep_fashion_dataset()
        train_dataset = ImageDataset(np.array(train_data))
        test_dataset = ImageDataset(np.array(test_data))
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=bn, num_workers=4)

        # Make Training
        with torch.autograd.set_detect_anomaly(False):
            for epoch in range(epochs+1):
                # Train on Train Set
                model.train()
                model.mode = 'train'
                for step, original in enumerate(train_loader):
                    original = original.to(device)
                    # Make transformations
                    tps_param_dic = tps_parameters(original.shape[0], arg.scal, arg.tps_scal, arg.rot_scal,
                                                   arg.off_scal, arg.scal_var, arg.augm_scal)
                    coord, vector = make_input_tps_param(tps_param_dic)
                    coord, vector = coord.to(device), vector.to(device)
                    image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                         original.shape[3], device)
                    image_appearance_t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(original)
                    image_spatial_t, image_appearance_t = normalize(image_spatial_t), normalize(image_appearance_t)
                    reconstruction, loss, rec_loss, equiv_loss, mu, L_inv = model(original, image_spatial_t,
                                                                                  image_appearance_t, coord, vector)
                    mu_norm = torch.mean(torch.norm(mu, p=1, dim=2)).cpu().detach().numpy()
                    L_inv_norm = torch.mean(torch.linalg.norm(L_inv, ord='fro', dim=[2, 3])).cpu().detach().numpy()
                    wandb.log({"Part Means": mu_norm})
                    wandb.log({"Precision Matrix": L_inv_norm})
                    # Zero out gradients
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # Track Loss
                    if step == 0:
                        loss_log = torch.tensor([loss])
                        rec_loss_log = torch.tensor([rec_loss])
                    else:
                        loss_log = torch.cat([loss_log, torch.tensor([loss])])
                        rec_loss_log = torch.cat([rec_loss_log, torch.tensor([rec_loss])])
                    training_loss = torch.mean(loss_log)
                    training_rec_loss = torch.mean(rec_loss_log)
                    wandb.log({"Training Loss": training_loss})
                    wandb.log({"Training Rec Loss": training_rec_loss})
                print(f'Epoch: {epoch}, Train Loss: {training_loss}')

                # Evaluate on Test Set
                model.eval()
                for step, original in enumerate(test_loader):
                    with torch.no_grad():
                        original = original.to(device)
                        tps_param_dic = tps_parameters(original.shape[0], arg.scal, arg.tps_scal, arg.rot_scal, arg.off_scal,
                                                       arg.scal_var, arg.augm_scal)
                        coord, vector = make_input_tps_param(tps_param_dic)
                        coord, vector = coord.to(device), vector.to(device)
                        image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                             original.shape[3], device)
                        image_appearance_t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(original)
                        image_spatial_t, image_appearance_t = normalize(image_spatial_t), normalize(image_appearance_t)
                        reconstruction, loss, rec_loss, equiv_loss, mu, L_inv = model(original, image_spatial_t, image_appearance_t, coord, vector)
                        if step == 0:
                            loss_log = torch.tensor([loss])
                        else:
                            loss_log = torch.cat([loss_log, torch.tensor([loss])])
                evaluation_loss = torch.mean(loss_log)
                wandb.log({"Evaluation Loss": evaluation_loss})
                print(f'Epoch: {epoch}, Test Loss: {evaluation_loss}')

                # Track Progress
                if True:
                    model.mode = 'predict'
                    original, fmap_shape, fmap_app, reconstruction = model(original, image_spatial_t,
                                                                           image_appearance_t, coord, vector)
                    make_visualization(original, reconstruction, image_spatial_t, image_appearance_t,
                                       fmap_shape, fmap_app, model_save_dir, epoch, device)
                    save_model(model, model_save_dir)

    elif mode == 'predict':
        # Make Directory for Predictions
        model_save_dir = '../results/' + name
        if not os.path.exists(model_save_dir + '/predictions'):
            os.makedirs(model_save_dir + '/predictions')
        # Load Model and Dataset
        model = Model(arg).to(device)
        model = load_model(model, model_save_dir).to(device)
        data = load_deep_fashion_dataset()
        test_data = np.array(data[-4:])
        test_dataset = ImageDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=bn)
        model.mode = 'predict'
        model.eval()
        # Predict on Dataset
        for step, original in enumerate(test_loader):
            with torch.no_grad():
                original = original.to(device)
                tps_param_dic = tps_parameters(original.shape[0], arg.scal, arg.tps_scal, arg.rot_scal, arg.off_scal,
                                               arg.scal_var, arg.augm_scal)
                coord, vector = make_input_tps_param(tps_param_dic)
                coord, vector = coord.to(device), vector.to(device)
                image_spatial_t, _ = ThinPlateSpline(original, coord, vector,
                                                     original.shape[3], device)
                image_appearance_t = K.ColorJitter(arg.brightness, arg.contrast, arg.saturation, arg.hue)(original)
                image, reconstruction, mu, shape_stream_parts, heat_map = model(original, image_spatial_t,
                                                                                image_appearance_t, coord, vector)


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)

