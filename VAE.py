import torch
import torch.nn as nn
from ops import get_heat_map, fold_img_with_L_inv
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
import torch.nn.functional as F
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, keypoint_metric
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from utils import count_parameters, visualize_VAE


def softmax(logit_map):
    bn, kn, h, w = logit_map.shape
    map_norm = nn.Softmax(dim=2)(logit_map.reshape(bn, kn, -1)).reshape(bn, kn, h, w)
    return map_norm


def coordinate_transformation(coords, grid, device, grid_size=64):
    bn, k, _ = coords.shape
    bucket = torch.linspace(-1., 1., grid_size, device=device)
    indices = torch.bucketize(coords.contiguous(), bucket)
    indices = indices.unsqueeze(-2).unsqueeze(-2)
    grid = grid.unsqueeze(1).repeat(1, k, 1, 1, 1)
    new_coords = torch.gather(grid, 3, indices).squeeze(-2).squeeze(-2)

    return new_coords


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.LeakyReLU()
        if bn:
            self.bn = nn.InstanceNorm2d(out_dim)

        # self.initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, bn=False, relu=False)
        self.bn2 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, bn=False, relu=False)
        self.bn3 = nn.InstanceNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, bn=False, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, bn=False, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f):
        super(Hourglass, self).__init__()
        self.up1 = Residual(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = Residual(f, f)
        self.n = n
        # Recursive hourglass
        if self.n > 0:
            self.low2 = Hourglass(n - 1, f)
        else:
            self.low2 = Residual(f, f)
        self.low3 = Residual(f, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class PreProcessor(nn.Module):
    def __init__(self, residual_dim, reconstr_dim):
        super(PreProcessor, self).__init__()
        if reconstr_dim == 128:
            self.preprocess = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                  Residual(64, 128),
                                                  Residual(128, 128),
                                                  Residual(128, residual_dim),
                                                  )
        elif reconstr_dim == 256:
            self.preprocess = nn.Sequential(Conv(3, 64, kernel_size=6, stride=2, bn=True, relu=True),
                                                  Residual(64, 128),
                                                  nn.MaxPool2d(2, 2),
                                                  Residual(128, 128),
                                                  Residual(128, residual_dim),
                                                  )
    def forward(self, img):
        img_preprocess = self.preprocess(img)

        return img_preprocess


class PartGenerator(nn.Module):
    def __init__(self, k, c, residual_dim, device, dim=64):
        super(PartGenerator, self).__init__()
        self.k = k
        self.c = c
        self.dim = dim
        self.to_channels = Residual(residual_dim, c * k)
        self.mu = nn.Conv2d(in_channels=c * k, out_channels=2 * k, kernel_size=dim, groups=k)
        self.prec = nn.Conv2d(in_channels=c * k, out_channels=4 * k, kernel_size=dim, groups=k)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.bn = nn.InstanceNorm2d(c * k)
        self.to_residual = Residual(c * k, residual_dim)
        self.device = device

    def forward(self, img_hg):
        bn = img_hg.shape[0]
        img_c = self.to_channels(img_hg)
        img_c = self.relu(self.bn(img_c))

        mu = self.mu(img_c).reshape(bn, self.k, 2)
        # mu = self.tanh(mu)
        prec = self.prec(img_c).reshape(bn, self.k, 2, 2)
        heat_map = get_heat_map(mu, prec, device=self.device, h=self.dim)
        # heat_map = softmax(heat_map)
        norm = torch.sum(heat_map, 1, keepdim=True) + 1
        heat_map_norm = heat_map / norm

        img_c = img_c.reshape(bn, self.k, self.c, self.dim, self.dim)
        heat_map_norm_repeat = heat_map_norm.unsqueeze(2).repeat(1, 1, self.c, 1, 1)
        confined = img_c * heat_map_norm_repeat
        confined = confined.reshape(bn, -1, self.dim, self.dim)
        confined = self.to_residual(confined)

        return mu, prec, confined, heat_map_norm


class Encoder(nn.Module):
    def __init__(self, k, c, residual_dim, n_features, depth_s, depth_a, device):
        super(Encoder, self).__init__()
        self.E_sigma = Hourglass(depth_s, residual_dim)
        self.E_alpha = Hourglass(depth_a, residual_dim)
        self.to_raw_features = Residual(residual_dim, n_features)
        self.PartGenerator = PartGenerator(k, c, residual_dim, device)

    def forward(self, img_preprocess):
        bn = img_preprocess.shape[0]
        img_preprocess_ss = img_preprocess[:bn // 2]
        img_preprocess_as = img_preprocess[bn // 2:]

        # Shape Stream
        img_hg_ss = self.E_sigma(img_preprocess_ss)
        mu_ss, prec_ss, confined_ss, heat_map_norm_ss = self.PartGenerator(img_hg_ss)

        # Appearance Stream
        img_hg_as = self.E_sigma(img_preprocess_as)
        mu_as, prec_as, confined_as, heat_map_norm_as = self.PartGenerator(img_hg_as)
        raw_features = self.E_alpha(confined_as)
        raw_features = self.to_raw_features(raw_features)

        # get features
        features = torch.einsum('bfij, bkij -> bkf', raw_features, heat_map_norm_ss)
        heat_feature_map = torch.einsum('bkij,bkn -> bnij', heat_map_norm_ss, features)

        return mu_ss, prec_ss, mu_as, prec_as, heat_feature_map, heat_map_norm_ss


class Decoder(nn.Module):
    def __init__(self, k, c, residual_dim, reconstr_dim, n_features, depth, dim=64):
        super(Decoder, self).__init__()
        self.c = c
        self.k = k
        self.reconstr_dim = reconstr_dim
        self.dim = dim
        self.to_residual1 = Residual(n_features, residual_dim)
        self.to_residual2 = Residual(c * k, residual_dim)
        self.hg = Hourglass(depth, residual_dim)
        self.to_channels = Residual(residual_dim, c * k)
        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.InstanceNorm2d(residual_dim)
        self.bn2 = nn.InstanceNorm2d(residual_dim // 2)
        self.bn3 = nn.InstanceNorm2d(residual_dim // 4)
        self.up_Conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=residual_dim, out_channels=residual_dim // 2,
                                                          kernel_size=4, stride=2, padding=1),
                                       self.bn2,
                                       self.relu)
        self.up_Conv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=residual_dim // 2, out_channels=residual_dim // 4,
                                                          kernel_size=4, stride=2, padding=1),
                                       self.bn3,
                                       self.relu)

        if reconstr_dim == 128:
            self.to_rgb = Conv(residual_dim // 2, 3, kernel_size=5, stride=1, bn=False, relu=False)
        else:
            self.to_rgb = Conv(residual_dim // 4, 3, kernel_size=5, stride=1, bn=False, relu=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, heat_feature_map, heat_map):
        bn = heat_feature_map.shape[0]
        heat_feature_map_residual = self.to_residual1(heat_feature_map)
        out_hg = self.hg(heat_feature_map_residual)

        # out_to_channels = self.to_channels(out_hg)
        # out_to_channels = out_to_channels.reshape(bn, self.k, self.c, self.dim, self.dim)
        # heat_maprepeat = heat_map.unsqueeze(2).repeat(1, 1, self.c, 1, 1)
        # confined = out_to_channels * heat_maprepeat
        # confined = confined.reshape(bn, -1, self.dim, self.dim)
        # confined = self.to_residual2(confined)
        # confined = self.relu(self.bn1(confined))
        heat_map_overlay = torch.sum(heat_map, dim=1).unsqueeze(1)
        confined = out_hg * heat_map_overlay

        up1 = self.up_Conv1(confined)
        if self.reconstr_dim == 256:
            up2 = self.up_Conv2(up1)
            reconstruction = self.to_rgb(up2)
        else:
            reconstruction = self.to_rgb(up1)
        reconstruction = self.sigmoid(reconstruction)

        return reconstruction


class VAE(nn.Module):
    def __init__(self, k, n_features, c, residual_dim, reconstr_dim, depth_s, depth_a, device):
        super(VAE, self).__init__()
        self.PreProcessor = PreProcessor(residual_dim, reconstr_dim)
        self.encoder = Encoder(k, c, residual_dim, n_features, depth_s, depth_a, device)
        self.decoder = Decoder(k, c, residual_dim, reconstr_dim, n_features, depth_a)

    def forward(self, img):
        img_preprocess = self.PreProcessor(img)
        mu_ss, prec_ss, mu_as, prec_as, heat_feature_map, heat_map_norm_ss = self.encoder(img_preprocess)
        reconstruction = self.decoder(heat_feature_map, heat_map_norm_ss)

        return mu_ss, prec_ss, mu_as, prec_as, heat_map_norm_ss, reconstruction


class VAE_Model(nn.Module):
    def __init__(self, arg):
        super(VAE_Model, self).__init__()
        self.scal = arg.scal
        self.tps_scal = arg.tps_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.k = arg.n_parts
        self.n_features = arg.n_features
        self.residual_dim = arg.residual_dim
        self.reconstr_dim = arg.reconstr_dim
        self.depth_s = arg.depth_s
        self.depth_a = arg.depth_a
        self.L_mu = arg.L_mu
        self.L_rec = arg.L_rec
        self.L_cov = arg.L_cov
        self.device = arg.device
        self.vae = VAE(self.k, self.n_features, 16, self.residual_dim, self.reconstr_dim, self.depth_s,
                       self.depth_a, self.device)

    def forward(self, img):
        # Make Transformation
        batch_size = img.shape[0]
        tps_param_dic = tps_parameters(batch_size, self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector = make_input_tps_param(tps_param_dic)
        coord, vector = coord.to(self.device), vector.to(self.device)
        img_rot, mesh_rot = ThinPlateSpline(img, coord, vector, self.reconstr_dim, device=self.device)

        # Send Stack through VAE
        img_stack = torch.cat((img, img_rot), dim=0)
        mu_ss, prec_ss, mu_as, prec_as, heat_map_norm_ss, reconstruction = self.vae(img_stack)

        #Sample from the Latent Space for Equivariance Loss
        # distribution = torch.distributions.multivariate_normal.MultivariateNormal(mu, precision_matrix=prec)
        # samples = distribution.sample()[:, :-1].to(self.device) # Dont use background map
        # samples1, samples2 = samples[:batch_size], samples[batch_size:]
        # samples1_rot = coordinate_transformation(samples1, mesh_rot, self.device)

        # Calculate Loss
        # Penalize mu values > 1. (outside image)
        out_loss = torch.mean(torch.max(torch.abs(mu_ss) - 1., torch.zeros_like(mu_ss)) + torch.max(torch.abs(mu_as) - 1., torch.zeros_like(mu_as)))
        mu_loss = torch.mean((mu_ss - mu_as) ** 2)
        eps = 1e-7
        precision_sq = (1 / prec_ss - 1 / prec_as) ** 2
        prec_loss = torch.mean(torch.sqrt(torch.sum(precision_sq, dim=[2, 3]) + eps))
        rec_loss = nn.BCELoss()(reconstruction, img)
        total_loss = self.L_mu * mu_loss + self.L_rec * rec_loss + self.L_cov * prec_loss + 500 * out_loss

        return img_rot, reconstruction, mu_ss, prec_ss, heat_map_norm_ss, total_loss


def main(arg):
    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(42)
    rng = np.random.RandomState(42)

    # Get args
    bn = arg.batch_size
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
    epochs = arg.epochs
    device = torch.device('cuda:' + str(arg.gpu) if torch.cuda.is_available() else 'cpu')
    arg.device = device

    # Load Datasets and DataLoader
    dataset = get_dataset(arg.dataset)
    if arg.dataset == 'pennaction':
        init_dataset = dataset(size=arg.reconstr_dim, action_req=["tennis_serve", "tennis_forehand", "baseball_pitch",
                                                                  "baseball_swing", "jumping_jacks", "golf_swing"])
        splits = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
        train_dataset, test_dataset = torch.utils.data.random_split(init_dataset, splits)
    else:
        train_dataset = dataset(size=arg.reconstr_dim, train=True)
        test_dataset = dataset(size=arg.reconstr_dim, train=False)
    train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=bn, shuffle=True, num_workers=4)

    if mode == 'train':
        # Make new directory
        model_save_dir = '../results/' + name
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/summary')

        # Save Hyperparameters
        write_hyperparameters(arg.toDict(), model_save_dir)

        # Define Model
        model = VAE_Model(arg).to(device)
        if load_from_ckpt:
            model = load_model(model, model_save_dir, device).to(device)
        print(f'Number of Parameters: {count_parameters(model)}')

        # Definde Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Log with wandb
        wandb.init(project='Disentanglement', config=arg, name=arg.name)
        wandb.watch(model, log='all')

        # Make Training
        with torch.autograd.set_detect_anomaly(False):
            for epoch in range(epochs+1):
                # Train on Train Set
                model.train()
                model.mode = 'train'
                for step, (original, keypoints) in enumerate(train_loader):
                    original, keypoints = original.to(device), keypoints.to(device)
                    img_rot, reconstruction, mu, prec, heatmap, total_loss = model(original)
                    mu_norm = torch.mean(torch.norm(mu, p=1, dim=2)).cpu().detach().numpy()
                    L_inv_norm = torch.mean(torch.linalg.norm(prec, ord='fro', dim=[2, 3])).cpu().detach().numpy()
                    # Track Mean and Precision Matrix
                    wandb.log({"Part Means": mu_norm})
                    wandb.log({"Precision Matrix": L_inv_norm})
                    # Zero out gradients
                    optimizer.zero_grad()
                    total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), arg.clip)
                    optimizer.step()
                    # Track Loss
                    wandb.log({"Training Loss": total_loss})
                    # Track Metric
                    score = keypoint_metric(mu, keypoints)
                    wandb.log({"Metric Train": score})

                # Evaluate on Test Set
                model.eval()
                val_score = torch.zeros(1)
                val_loss = torch.zeros(1)
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        original, keypoints = original.to(device), keypoints.to(device)
                        img_rot, reconstruction, mu, prec, heatmap, total_loss = model(original)
                        # Track Loss and Metric
                        score = keypoint_metric(mu, keypoints)
                        val_score += score.cpu()
                        val_loss += total_loss.cpu()

                val_loss = val_loss / (step + 1)
                val_score = val_score / (step + 1)
                wandb.log({"Evaluation Loss": val_loss})
                wandb.log({"Metric Validation": val_score})

                # Track Progress & Visualization
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        model.mode = 'predict'
                        original, keypoints = original.to(device), keypoints.to(device)
                        img_rot, reconstruction, mu, prec, heatmap, total_loss = model(original)
                        img = visualize_VAE(original, img_rot, reconstruction, mu, heatmap, keypoints)
                        wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        save_model(model, model_save_dir)

                        if step == 0:
                            break

if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)





