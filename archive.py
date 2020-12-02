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