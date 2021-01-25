import torch
import torch.nn as nn
from ops import get_heat_map, augm, feat_mu_to_enc, fold_img_with_L_inv
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
import torch.nn.functional as F
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, keypoint_metric, visualize_SAE, count_parameters
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from architecture import Decoder as Decoder_old

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
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.LeakyReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

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
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, bn=False, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, bn=False, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
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
    def __init__(self, n_parts, n_features, residual_dim, reconstr_dim, depth_s, depth_a, device, p_dropout=0.2, c=16, dim=64):
        super(Encoder, self).__init__()
        self.preprocessor = PreProcessor(residual_dim, reconstr_dim)
        self.k = n_parts
        self.dim = dim
        self.dropout = nn.Dropout(p_dropout)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(c * self.k)
        self.bn2 = nn.BatchNorm2d(2 * n_parts)
        self.bn3 = nn.BatchNorm2d(4 * n_parts)
        self.hg_shape = Hourglass(depth_s, residual_dim)
        self.out = Conv(residual_dim, residual_dim, kernel_size=3, stride=1, bn=True, relu=True)
        self.to_parts = Conv(residual_dim, self.k, kernel_size=3, stride=1, bn=False, relu=False)
        self.map_transform = Conv(self.k, residual_dim, 3, 1, bn=False, relu=False)
        self.prec = nn.Conv2d(in_channels=self.k, out_channels=2 * self.k, kernel_size=dim, groups=self.k)
        # self.to_channels = Conv(residual_dim, c * n_parts, kernel_size=3, stride=1, bn=False, relu=False)
        # self.mu = nn.Conv2d(in_channels=c * n_parts, out_channels=2 * n_parts, kernel_size=dim, groups=n_parts)
        # self.prec = nn.Conv2d(in_channels=c * n_parts, out_channels=2 * n_parts, kernel_size=dim, groups=n_parts)
        # self.to_channels = Conv(residual_dim, self.k, kernel_size=3, stride=1, bn=True, relu=True)
        # self.mu = nn.Linear(self.k * 64 * 64, self.k * 2)
        # self.prec = nn.Linear(self.k * 64 * 64, self.k * 4)

        self.hg_appearance = Hourglass(depth_a, residual_dim)
        # self.to_residual = Conv(self.k, residual_dim, kernel_size=1, stride=1, bn=False, relu=False)
        self.to_features = Conv(residual_dim, n_features, kernel_size=1, stride=1, bn=False, relu=False)
        self.device = device

    def forward(self, img):
        bn = img.shape[0]
        img_preprocessed = self.preprocessor(img)

        # Get Shape Representation
        img_shape = self.hg_shape(img_preprocessed)
        img_shape = self.dropout(img_shape)
        img_shape = self.out(img_shape)
        feature_map = self.to_parts(img_shape)
        map_normalized = softmax(feature_map)
        mu = get_mu(map_normalized, self.device)

        # Get Stack for Appearance Hourglass
        map_transformed = self.map_transform(map_normalized)
        stack = map_transformed + img_preprocessed


        # img_c = self.to_channels(img_shape)
        # img_c = self.relu(self.bn1(img_c))
        # mu = self.mu(img_c)
        # mu = mu.reshape(bn, self.k, 2)

        prec = self.prec(feature_map)
        prec = self.sigmoid(prec).reshape(bn, self.k, 2)
        rot, scal = 3.141 * prec[:, :, 0].reshape(-1), 20 * prec[:, :, 1].reshape(-1)
        scal_matrix = torch.cat([torch.tensor([[scal[i], 0.], [0., 0.]], device=self.device).unsqueeze(0) for i in range(scal.shape[0])], 0).reshape(bn, self.k, 2, 2)
        rot_mat = torch.cat([rotation_mat(rot[i].reshape(-1)).unsqueeze(0) for i in range(rot.shape[0])], 0).reshape(bn, self.k, 2, 2)
        prec = torch.tensor([[30., 0.], [0., 30.]], device=self.device).unsqueeze(0).unsqueeze(0).repeat(bn, self.k, 1, 1) - \
               scal_matrix
        prec = rot_mat @ prec @ rot_mat.transpose(2, 3)

        heat_map = get_heat_map(mu, prec, device=self.device, h=self.dim)
        norm = torch.sum(heat_map, 1, keepdim=True) + 1
        heat_map_norm = heat_map / norm

        # Get Appearance Representation
        # heat_map_norm_res = self.to_residual(heat_map_norm)
        # img_stack = img_preprocessed + heat_map_norm_res
        img_app = self.hg_appearance(stack)

        raw_features = self.to_features(img_app)
        part_appearances = torch.einsum('bfij, bkij -> bkf', raw_features, heat_map_norm)

        return mu, prec, map_normalized, heat_map_norm, part_appearances


class Decoder(nn.Module):
    def __init__(self, n_features, residual_dim, reconstr_dim, depth_s, device, nk, covariance):
        super(Decoder, self).__init__()
        self.decoder_old = Decoder_old(nk, n_features, reconstr_dim)
        self.device = device
        self.reconstr_dim = reconstr_dim
        self.covariance = covariance
        # self.decoder = Hourglass(depth_s, residual_dim)
        # self.to_residual = Residual(n_features, residual_dim)
        # self.relu = nn.LeakyReLU()
        # self.bn1 = nn.BatchNorm2d(residual_dim // 2)
        # self.bn2 = nn.BatchNorm2d(residual_dim // 4)
        # self.up_Conv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=residual_dim, out_channels=residual_dim // 2,
        #                                                  kernel_size=4, stride=2, padding=1),
        #                               self.bn1,
        #                               self.relu)
        # self.up_Conv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=residual_dim // 2, out_channels=residual_dim // 4,
        #                                                  kernel_size=4, stride=2, padding=1),
        #                               self.bn2,
        #                               self.relu)
        #
        # if reconstr_dim == 128:
        #     self.to_rgb = Conv(residual_dim // 2, 3, kernel_size=5, stride=1, bn=False, relu=False)
        # else:
        #     self.to_rgb = Conv(residual_dim // 4, 3, kernel_size=5, stride=1, bn=False, relu=False)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, heat_map_norm, part_appearances, mu, prec):
        encoding = feat_mu_to_enc(part_appearances, mu, prec, self.device, self.covariance, self.reconstr_dim)
        reconstruction = self.decoder_old(encoding)
        # heat_feat_map = torch.einsum('bkij,bkn -> bnij', heat_map_norm, part_appearances)
        # heat_feat_map = self.to_residual(heat_feat_map)
        # out = self.decoder(heat_feat_map)
        # heat_map_overlay = torch.sum(heat_map_norm, dim=1).unsqueeze(1)
        # heat_map_mask = torch.where(heat_map_overlay > 0.2, torch.ones_like(heat_map_overlay), heat_map_overlay)
        # out_masked = out * heat_map_mask
        #
        # up1 = self.up_Conv1(out_masked)
        # if self.reconstr_dim == 256:
        #     up2 = self.up_Conv2(up1)
        #     reconstruction = self.to_rgb(up2)
        # else:
        #     reconstruction = self.to_rgb(up1)
        # reconstruction = self.sigmoid(reconstruction)

        return reconstruction


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
        self.L_mu = arg.L_mu
        self.L_rec = arg.L_rec
        self.L_cov = arg.L_cov
        self.L_conc = arg.L_conc
        self.device = arg.device
        self.encoder = Encoder(arg.n_parts, arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s,
                               arg.depth_a, self.device, p_dropout=arg.p_dropout)
        self.decoder = Decoder(arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s, self.device, self.k,
                               arg.covariance)

    def forward(self, img):
        # Make Transformation
        batch_size = img.shape[0]
        tps_param_dic = tps_parameters(batch_size, self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector, rot_mat = make_input_tps_param(tps_param_dic)
        coord, vector, rot_mat = coord.to(self.device), vector.to(self.device), rot_mat.to(self.device)
        rot_mat = rot_mat.unsqueeze(1).repeat(1, self.k, 1, 1)
        img_rot, mesh_rot = ThinPlateSpline(img, coord, vector, self.reconstr_dim, device=self.device)
        img_stack = torch.cat([img, img_rot])
        img_stack_augm = augm(img_stack, self.arg, self.device)
        img_augm, img_rot_augm = img_stack_augm[:batch_size], img_stack_augm[batch_size:]

        # Send through encoder
        mu_augm, prec_augm, part_map_augm, heat_map_norm_augm, part_appearances_augm = self.encoder(img_augm)
        mu_rot, prec_rot, part_map_rot, heat_map_norm_rot, part_appearances_rot = self.encoder(img_rot)

        # Send through decoder
        img_reconstr = self.decoder(heat_map_norm_augm, part_appearances_rot, mu_augm, prec_augm)
        # # img_reconstr_augm = self.decoder(heat_map_norm_augm, part_appearances_augm)
        # # img_reconstr_rot = self.decoder(heat_map_norm_rot, part_appearances_rot)
        img_reconstr_rot_augm = self.decoder(heat_map_norm_rot, part_appearances_augm, mu_rot, prec_rot)

        # Calculate Loss
        # mu_augm_rot = (mu_augm.unsqueeze(2) @ rot_mat).squeeze(2)
        # prec_augm_rot = rot_mat @ prec_augm @ rot_mat.transpose(2, 3)
        #
        # out_loss = torch.mean(torch.max(torch.abs(mu_augm) - 1., torch.zeros_like(mu_augm)) +
        #                       torch.max(torch.abs(mu_rot) - 1., torch.zeros_like(mu_rot)))
        # mu_loss = torch.mean((mu_augm_rot - mu_rot) ** 2)
        # prec_loss = torch.tensor([1.], device=self.device) - torch.mean(F.cosine_similarity(torch.flatten(prec_augm_rot, 2),
        #                                                                 torch.flatten(prec_rot, 2), dim=2))
        # conc_loss = 1 / torch.mean(torch.linalg.norm(prec_augm, ord='fro', dim=[2, 3]) + \
        #                            torch.linalg.norm(prec_rot, ord='fro', dim=[2, 3]))

        # Reconstruction loss
        rec_loss1 = nn.MSELoss()(img_reconstr, img)
        rec_loss4 = nn.MSELoss()(img_reconstr_rot_augm, img_rot_augm)
        rec_loss = rec_loss1 + rec_loss4
        # img_stack = torch.cat([img, img_rot_augm], dim=0)
        # rec_stack = torch.cat([img_reconstr, img_reconstr_rot_augm])
        # mu_stack = torch.cat([mu_augm, mu_rot])
        # prec_stack = torch.cat([prec_augm, prec_rot])
        # distance_metric = torch.abs(img_stack - rec_stack)
        # fold_img_squared = fold_img_with_L_inv(distance_metric, mu_stack.detach(), prec_stack.detach(),
        #                                        0.8, 0.2, self.device)
        # rec_loss = torch.mean(torch.sum(fold_img_squared, dim=[2, 3]))

        total_loss = self.L_mu * mu_loss + self.L_rec * rec_loss + self.L_cov * prec_loss + \
                     0 * conc_loss + 0 * out_loss

        return img, img_reconstr, img_reconstr, img_augm, img_reconstr, img_augm, img_rot_augm, \
               img_reconstr_rot_augm, heat_map_norm_augm, heat_map_norm_rot, mu_augm, prec_augm, total_loss


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
        model = SAE(arg).to(device)
        if load_from_ckpt:
            model = load_model(model, model_save_dir, device).to(device)
        print(f'Number of Parameters: {count_parameters(model)}')

        # Definde Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=arg.weight_decay)

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
                    img, img_reconstr, img_augm, img_reconstr_augm, img_rot, img_reconstr_rot, img_rot_augm, \
                    img_reconstr_rot_augm, heat_map_norm_augm, heat_map_norm_rot, mu, prec, total_loss = model(original)
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
                    wandb.log({"Training Loss": total_loss.cpu()})
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
                        img, img_reconstr, img_augm, img_reconstr_augm, img_rot, img_reconstr_rot, img_rot_augm, \
                        img_reconstr_rot_augm, heat_map_norm_augm, heat_map_norm_rot, mu, prec, total_loss = model(original)
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
                        img, img_reconstr, img_augm, img_reconstr_augm, img_rot, img_reconstr_rot, img_rot_augm, \
                        img_reconstr_rot_augm, heat_map_norm_augm, heat_map_norm_rot, mu, prec, total_loss = model(original)

                        img = visualize_SAE(img, img_reconstr, img_augm, img_reconstr_augm, img_rot, img_reconstr_rot, img_rot_augm,
                                            img_reconstr_rot_augm, heat_map_norm_augm, heat_map_norm_rot, mu, keypoints)
                        wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        save_model(model, model_save_dir)

                        if step == 0:
                            break

if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)
