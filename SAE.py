import torch
import torch.nn as nn
from ops import get_heat_map, augm, feat_mu_to_enc, fold_img_with_L_inv, AbsDetJacobian, loss_fn, get_mu_and_prec
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, keypoint_metric, visualize_SAE, count_parameters
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from architecture import Decoder as Decoder_old
from transformer import ViT
from LambdaNetworks import GSA_Transformer
from torch.utils.data import ConcatDataset, random_split


def coordinate_transformation(coords, grid, device, grid_size=1000):
    bn, k, _ = coords.shape
    bucket = torch.linspace(-10., 10., grid_size, device=device)
    indices = torch.bucketize(coords.contiguous(), bucket)
    indices = indices.unsqueeze(-2).unsqueeze(-2)
    grid = grid.unsqueeze(1).repeat(1, k, 1, 1, 1)
    new_coords = torch.gather(grid, 3, indices).squeeze(-2).squeeze(-2)

    return new_coords


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


def softmax(logit_map):
    bn, kn, h, w = logit_map.shape
    map_norm = nn.Softmax(dim=2)(logit_map.reshape(bn, kn, -1)).reshape(bn, kn, h, w)
    return map_norm


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=True, relu=True):
        super(Conv, self).__init__()
        self.kernel_size = kernel_size
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.LeakyReLU()
        if bn:
            self.bn = nn.InstanceNorm2d(out_dim)

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


class Encoder(nn.Module):
    def __init__(self, n_parts, n_features, residual_dim, reconstr_dim, depth_s, depth_a, background, device,
                 p_dropout, t_patch_size, t_dim, t_depth, t_heads, t_mlp_dim, c=16, dim=64):
        super(Encoder, self).__init__()
        self.preprocessor = PreProcessor(residual_dim, reconstr_dim)
        self.k = n_parts + 1 if background else n_parts
        self.dim = dim
        self.bn = nn.BatchNorm2d(2 * self.k)
        self.sigmoid = nn.Sigmoid()
        self.map_transform = Conv(self.k, residual_dim, 3, 1, bn=False, relu=False)

        self.prec = nn.Conv2d(in_channels=self.k, out_channels=2 * self.k, kernel_size=dim, groups=self.k)

        self.background = background
        self.device = device

        # Hourglass Shape
        # self.hg_shape = Hourglass(depth_s, residual_dim)
        # self.dropout = nn.Dropout(p_dropout)
        # self.out = Conv(residual_dim, residual_dim, kernel_size=3, stride=1, bn=True, relu=True)
        # self.to_parts = Conv(residual_dim, self.k, kernel_size=3, stride=1, bn=False, relu=False)

        # Hourglass Appearance
        # self.hg_appearance = Hourglass(depth_a, residual_dim)
        # self.to_features = Conv(residual_dim, n_features, kernel_size=1, stride=1, bn=False, relu=False)

        # Transformer Shape
        self.conv1 = Conv(residual_dim, residual_dim, kernel_size=3, stride=1, bn=True, relu=True)
        self.vit_shape = ViT(
                       image_size=64,
                       patch_size=t_patch_size,
                       dim=t_dim,
                       depth=t_depth,
                       heads=t_heads,
                       mlp_dim=t_mlp_dim,
                       dropout=0.1,
                       channels=256,
                       emb_dropout=0.1,
                       nk=self.k
                       )

        self.vit_appearance = ViT(
            image_size=64,
            patch_size=t_patch_size,
            dim=t_dim,
            depth=t_depth,
            heads=t_heads,
            mlp_dim=t_mlp_dim,
            dropout=0.1,
            channels=256,
            emb_dropout=0.1,
            nk=n_features
        )

    def forward(self, img):
        bn = img.shape[0]
        img_preprocessed = self.preprocessor(img)

        # Get Shape Representation
        # img_shape = self.hg_shape(img_preprocessed)
        # img_shape = self.dropout(img_shape)
        # img_shape = self.out(img_shape)
        # feature_map = self.to_parts(img_shape)

        # Transformer
        img_shape = self.conv1(img_preprocessed)
        feature_map = self.vit_shape(img_shape)

        map_normalized = softmax(feature_map)
        mu = get_mu(map_normalized, self.device)

        # Get Stack for Appearance Hourglass
        map_transformed = self.map_transform(map_normalized)
        stack = map_transformed + img_preprocessed

        # Predict precision matrix
        prec = self.prec(feature_map)
        prec = self.sigmoid(self.bn(prec)).reshape(bn, self.k, 2)
        rot, scal = 2 * 3.141 * prec[:, :, 0].reshape(-1), 20 * prec[:, :, 1].reshape(-1)
        scal_matrix = torch.cat([torch.tensor([[scal[i], 0.], [0., 0.]], device=self.device).unsqueeze(0) for i in range(scal.shape[0])], 0).reshape(bn, self.k, 2, 2)
        rot_mat = torch.cat([rotation_mat(rot[i].reshape(-1)).unsqueeze(0) for i in range(rot.shape[0])], 0).reshape(bn, self.k, 2, 2)
        prec = torch.tensor([[30., 0.], [0., 30.]], device=self.device).unsqueeze(0).unsqueeze(0).repeat(bn, self.k, 1, 1) - \
               scal_matrix
        prec = rot_mat @ prec @ rot_mat.transpose(2, 3)
        if self.background:
            eps = 1e-6
            prec[:, -1] = torch.tensor([[0.01, eps], [eps, 0.01]], device=self.device)

        # Make Heatmap
        heat_map = get_heat_map(mu, prec, self.device, self.background, self.dim)
        norm = torch.sum(heat_map, 1, keepdim=True) + 1
        heat_map_norm = heat_map / norm
        if self.background:
            heat_map_norm[:, -1] = 1 / heat_map_norm[:, -1]

        # Get Appearance Representation
        # img_app = self.hg_appearance(stack)
        # raw_features = self.to_features(img_app)

        # Transformer
        raw_features = self.vit_appearance(stack)

        part_appearances = torch.einsum('bfij, bkij -> bkf', raw_features, heat_map_norm)

        return mu, prec, map_normalized, heat_map_norm, part_appearances


class Decoder(nn.Module):
    def __init__(self, n_features, residual_dim, reconstr_dim, depth_s, device, nk, covariance, background):
        super(Decoder, self).__init__()
        self.k = nk + 1 if background else nk
        self.device = device
        self.reconstr_dim = reconstr_dim
        self.covariance = covariance
        self.background = background
        self.decoder_old = Decoder_old(self.k, n_features, reconstr_dim)

        # self.vit_decoder = ViT(
        #     image_size=64,
        #     patch_size=4,
        #     dim=256,
        #     depth=8,
        #     heads=8,
        #     mlp_dim=1024,
        #     dropout=0.1,
        #     channels=n_features,
        #     emb_dropout=0.1,
        #     nk=256
        # )
        # self.relu = nn.ReLU()
        # self.bn1 = nn.InstanceNorm2d(128)
        # self.bn2 = nn.InstanceNorm2d(64)
        # self.up_Conv1 = nn.Sequential(
        #                               nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,
        #                                                  stride=2, padding=1),
        #                               self.bn1,
        #                               self.relu)
        #
        # self.up_Conv2 = nn.Sequential(
        #                               nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,
        #                                                  stride=2, padding=1),
        #                               self.bn2,
        #                               self.relu)
        # if self.reconstr_dim == 256:
        #     self.to_rgb = Conv(64, 3, kernel_size=3, stride=1, bn=False, relu=False)
        # else:
        #     self.to_rgb = Conv(128, 3, kernel_size=3, stride=1, bn=False, relu=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, heat_map_norm, part_appearances, mu, prec):
        encoding = feat_mu_to_enc(part_appearances, mu, prec, self.device, self.covariance, self.reconstr_dim, self.background)
        reconstruction = self.decoder_old(encoding)

        # With Transformer
        # encoding = torch.einsum('bkij, bkn -> bnij', heat_map_norm, part_appearances)
        # out = self.vit_decoder(encoding)
        # out = self.up_Conv1(out)
        # if self.reconstr_dim == 256:
        #     out = self.up_Conv2(out)
        # reconstruction = self.sigmoid(self.to_rgb(out))

        return reconstruction


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


class SAE(nn.Module):
    def __init__(self, arg):
        super(SAE, self).__init__()
        self.arg = arg
        self.k = arg.n_parts
        self.reconstr_dim = arg.reconstr_dim
        self.scal = arg.scal
        self.tps_scal = arg.tps_scal
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.fold_with_shape = arg.fold_with_shape
        self.background = arg.background
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.L_mu = arg.L_mu
        self.L_rec = arg.L_rec
        self.L_cov = arg.L_cov
        self.L_conc = arg.L_conc
        self.L_sep = arg.L_sep
        self.sig_sep = arg.sig_sep
        self.device = arg.device
        self.encoder = Encoder(arg.n_parts, arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s,
                               arg.depth_a, self.background, self.device, arg.p_dropout,
                               arg.t_patch_size, arg.t_dim, arg.t_depth, arg.t_heads, arg.t_mlp_dim)
        self.decoder = Decoder(arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s, self.device, self.k,
                               arg.covariance, self.background)

    def forward(self, img):
        bn = img.shape[0]
        # Make Transformation
        input_images, ground_truth_images, mesh_stack = make_pairs(img, self.arg)
        transform_mesh = F.interpolate(mesh_stack, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh, self.device)

        # Send through encoder
        mu, prec, part_map_norm, heat_map_norm, part_appearances = self.encoder(input_images)

        # Swap part appearances
        part_appearances_swap = torch.cat([part_appearances[bn:], part_appearances[:bn]], dim=0)

        # Send through decoder
        img_reconstr = self.decoder(heat_map_norm, part_appearances_swap, mu, prec)

        # Calculate Loss
        integrant = (part_map_norm.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
        mu_t = torch.einsum('akij, alij -> akl', integrant, transform_mesh)
        transform_mesh_out_prod = torch.einsum('amij, anij -> amnij', transform_mesh, transform_mesh)
        mu_out_prod = torch.einsum('akm, akn -> akmn', mu_t, mu_t)
        stddev_t = torch.einsum('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod

        total_loss, rec_loss, transform_loss, precision_loss = loss_fn(bn, mu, prec, mu_t, stddev_t,
                                                                       img_reconstr, ground_truth_images,
                                                                       self.fold_with_shape,
                                                                       self.l_2_scal, self.l_2_threshold, self.L_mu,
                                                                       self.L_cov,
                                                                       self.L_rec, self.L_sep, self.sig_sep,
                                                                       self.background, self.device)
        if self.background:
            mu, prec = mu[:, :-1], prec[:, :-1]
        return ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm, total_loss


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
    if arg.dataset != "mix":
        dataset = get_dataset(arg.dataset)
    if arg.dataset == 'pennaction':
        init_dataset = dataset(size=arg.reconstr_dim, action_req=["tennis_serve", "tennis_forehand", "baseball_pitch",
                                                                  "baseball_swing", "jumping_jacks", "golf_swing"])
        splits = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
        train_dataset, test_dataset = random_split(init_dataset, splits, generator=torch.Generator().manual_seed(42))
    elif arg.dataset =='deepfashion':
        train_dataset = dataset(size=arg.reconstr_dim, train=True)
        test_dataset = dataset(size=arg.reconstr_dim, train=False)
    elif arg.dataset == 'human36':
        init_dataset = dataset(size=arg.reconstr_dim)
        splits = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
        train_dataset, test_dataset = random_split(init_dataset, splits, generator=torch.Generator().manual_seed(42))
    elif arg.dataset == 'mix':
        # add pennaction
        dataset_pa = get_dataset("pennaction")
        init_dataset_pa = dataset_pa(size=arg.reconstr_dim, action_req=["tennis_serve", "tennis_forehand", "baseball_pitch",
                                                                  "baseball_swing", "jumping_jacks", "golf_swing"], mix=True)
        splits_pa = [int(len(init_dataset_pa) * 0.8), len(init_dataset_pa) - int(len(init_dataset_pa) * 0.8)]
        train_dataset_pa, test_dataset_pa = random_split(init_dataset_pa, splits_pa, generator=torch.Generator().manual_seed(42))
        # add deepfashion
        dataset_df = get_dataset("deepfashion")
        train_dataset_df = dataset_df(size=arg.reconstr_dim, train=True, mix=True)
        test_dataset_df = dataset_df(size=arg.reconstr_dim, train=False, mix=True)
        # add human36
        dataset_h36 = get_dataset("human36")
        init_dataset_h36 = dataset_h36(size=arg.reconstr_dim, mix=True)
        splits_h36 = [int(len(init_dataset_h36) * 0.8), len(init_dataset_h36) - int(len(init_dataset_h36) * 0.8)]
        train_dataset_h36, test_dataset_h36 = random_split(init_dataset_h36, splits_h36, generator=torch.Generator().manual_seed(42))
        # Concatinate all
        train_datasets = [train_dataset_df, train_dataset_h36]
        test_datasets = [test_dataset_df, test_dataset_h36]
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)

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
        model = SAE(arg).to(device)
        if load_from_ckpt:
            model = load_model(model, model_save_dir, device).to(device)
        print(f'Number of Parameters: {count_parameters(model)}')

        # Definde Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=arg.weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, factor=0.2, threshold=1e-3, patience=3)

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
                    bn = original.shape[0]
                    original, keypoints = original.to(device), keypoints.to(device)
                    # Forward Pass
                    ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm, total_loss = model(original)
                    # Track Mean and Precision Matrix
                    mu_norm = torch.mean(torch.norm(mu[:bn], p=1, dim=2)).cpu().detach().numpy()
                    L_inv_norm = torch.mean(torch.linalg.norm(prec[:bn], ord='fro', dim=[2, 3])).cpu().detach().numpy()
                    wandb.log({"Part Means": mu_norm})
                    wandb.log({"Precision Matrix": L_inv_norm})
                    # Zero out gradients
                    optimizer.zero_grad()
                    total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), arg.clip)
                    optimizer.step()
                    # Track Loss
                    wandb.log({"Training Loss": total_loss.cpu()})
                    # Track Metric
                    score = keypoint_metric(mu[:bn], keypoints)
                    wandb.log({"Metric Train": score})

                # Evaluate on Test Set
                model.eval()
                val_score = torch.zeros(1)
                val_loss = torch.zeros(1)
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        bn = original.shape[0]
                        original, keypoints = original.to(device), keypoints.to(device)
                        ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm, total_loss= model(original)
                        # Track Loss and Metric
                        score = keypoint_metric(mu[:bn], keypoints)
                        val_score += score.cpu()
                        val_loss += total_loss.cpu()

                val_loss = val_loss / (step + 1)
                val_score = val_score / (step + 1)
                # scheduler.step()
                wandb.log({"Evaluation Loss": val_loss})
                wandb.log({"Metric Validation": val_score})

                # Track Progress & Visualization
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        model.mode = 'predict'
                        original, keypoints = original.to(device), keypoints.to(device)
                        ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm, total_loss = model(original)

                        img = visualize_SAE(ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm,
                                            keypoints, model_save_dir + '/summary/', epoch)
                        wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        save_model(model, model_save_dir)

                        if step == 0:
                            break

if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)
