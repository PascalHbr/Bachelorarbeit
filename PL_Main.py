import torch
import pytorch_lightning as pl
from PL_architecture import Encoder, Decoder
import wandb
from PL_ops import*
from utils import keypoint_metric
from Dataloader import DataLoader, get_dataset
from torch.utils.data import ConcatDataset, random_split
from config import parse_args, write_hyperparameters
from dotmap import DotMap
from pytorch_lightning.loggers import WandbLogger

class PLModel(pl.LightningModule):
    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.l_2_scal = arg.l_2_scal
        self.l_2_threshold = arg.l_2_threshold
        self.L_mu = arg.L_mu
        self.L_cov = arg.L_cov
        self.L_rec = arg.L_rec
        self.mode = arg.mode
        self.background = arg.background
        self.fold_with_L_inv = arg.fold_with_L_inv

        self.n_parts = arg.n_parts + 1 if self.background else arg.n_parts
        self.encoder = Encoder(arg.n_parts, arg.n_features, arg.residual_dim, arg.reconstr_dim, arg.depth_s, arg.depth_a,
                               arg.p_dropout, arg.hg_patch_size, arg.hg_dim, arg.hg_depth, arg.hg_heads, arg.hg_mlp_dim,
                               arg.module, arg.background)
        self.decoder = Decoder(arg.n_features, arg.reconstr_dim, arg.n_parts,
                               arg.dec_patch_size, arg.dec_dim, arg.dec_depth, arg.dec_heads, arg.dec_mlp_dim,
                               arg.module, arg.background)



    def forward(self, img):
        bn = img.shape[0]
        # Make Transformation
        input_images, ground_truth_images, mesh_stack = make_pairs(img, self.arg)
        transform_mesh = F.interpolate(mesh_stack, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh)

        # Send through encoder
        mu, L_inv, part_map_norm, heat_map, heat_map_norm, part_appearances = self.encoder(input_images)

        # Swap part appearances
        part_appearances_swap = torch.cat([part_appearances[bn:], part_appearances[:bn]], dim=0)

        # Send through decoder
        img_reconstr = self.decoder(heat_map_norm, part_appearances_swap, mu, L_inv)

        # Calculate Loss
        integrant = (part_map_norm.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
        mu_t = torch.einsum('akij, alij -> akl', integrant, transform_mesh)
        transform_mesh_out_prod = torch.einsum('amij, anij -> amnij', transform_mesh, transform_mesh)
        mu_out_prod = torch.einsum('akm, akn -> akmn', mu_t, mu_t)
        stddev_t = torch.einsum('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod

        loss = loss_fn(bn, mu, L_inv, mu_t, stddev_t, img_reconstr,
                                                                       ground_truth_images, self.l_2_scal,
                                                                       self.l_2_threshold,
                                                                       self.L_mu, self.L_cov, self.L_rec,
                                                                       self.background, self.fold_with_L_inv)
        if self.background:
            mu, L_inv = mu[:, :-1], L_inv[:, :-1]

        return ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.arg.lr, weight_decay=self.arg.weight_decay)
        return optimizer

    def training_step(self, batch, batch_index):
        original, keypoints = batch
        bn = original.shape[0]
        # Forward Pass
        ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, loss = self(
            original)
        # Track Mean and Precision Matrix
        mu_norm = torch.mean(torch.norm(mu[:bn], p=1, dim=2)).cpu().detach().numpy()
        L_inv_norm = torch.mean(torch.linalg.norm(L_inv[:bn], ord='fro', dim=[2, 3])).cpu().detach().numpy()
        self.log("Part Means", mu_norm)
        self.log("Precision Matrix", L_inv_norm)
        # # Track Loss
        self.log("Training Loss", loss.cpu())
        # # Track Metric
        score, mu, L_inv, part_map_norm, heat_map = keypoint_metric(mu, keypoints, L_inv,
                                                                    part_map_norm, heat_map, self.arg.reconstr_dim)
        self.log("Metric Train", score)

        return {'loss': loss, 'metric':score}

    def validation_step(self, batch, batch_index):
        results = self.training_step(batch, batch_index)
        return results

    def validation_step_end(self, val_step_outputs):
        avg_val_metric = torch.tensor([x['metric'] for x in val_step_outputs]).mean()
        return {'val_loss': avg_val_metric}

    def setup(self, stage=None):
        if self.arg.dataset != "mix":
            dataset = get_dataset(self.arg.dataset)
        if self.arg.dataset == 'pennaction':
            init_dataset = dataset(size=self.arg.reconstr_dim,
                                   action_req=["tennis_serve", "tennis_forehand", "baseball_pitch",
                                               "baseball_swing", "jumping_jacks", "golf_swing"])
            splits = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
            self.train_dataset, self.test_dataset = random_split(init_dataset, splits,
                                                       generator=torch.Generator().manual_seed(42))
        elif self.arg.dataset == 'deepfashion':
            self.train_dataset = dataset(size=self.arg.reconstr_dim, train=True)
            self.test_dataset = dataset(size=self.arg.reconstr_dim, train=False)
        elif self.arg.dataset == 'human36':
            init_dataset = dataset(size=self.arg.reconstr_dim)
            splits = [int(len(init_dataset) * 0.8), len(init_dataset) - int(len(init_dataset) * 0.8)]
            self.train_dataset, self.test_dataset = random_split(init_dataset, splits,
                                                       generator=torch.Generator().manual_seed(42))
        elif self.arg.dataset == 'mix':
            # add pennaction
            dataset_pa = get_dataset("pennaction")
            init_dataset_pa = dataset_pa(size=self.arg.reconstr_dim,
                                         action_req=["tennis_serve", "tennis_forehand", "baseball_pitch",
                                                     "baseball_swing", "jumping_jacks", "golf_swing"], mix=True)
            splits_pa = [int(len(init_dataset_pa) * 0.8), len(init_dataset_pa) - int(len(init_dataset_pa) * 0.8)]
            train_dataset_pa, test_dataset_pa = random_split(init_dataset_pa, splits_pa,
                                                             generator=torch.Generator().manual_seed(42))
            # add deepfashion
            dataset_df = get_dataset("deepfashion")
            train_dataset_df = dataset_df(size=self.arg.reconstr_dim, train=True, mix=True)
            test_dataset_df = dataset_df(size=self.arg.reconstr_dim, train=False, mix=True)
            # add human36
            dataset_h36 = get_dataset("human36")
            init_dataset_h36 = dataset_h36(size=self.arg.reconstr_dim, mix=True)
            splits_h36 = [int(len(init_dataset_h36) * 0.8), len(init_dataset_h36) - int(len(init_dataset_h36) * 0.8)]
            train_dataset_h36, test_dataset_h36 = random_split(init_dataset_h36, splits_h36,
                                                               generator=torch.Generator().manual_seed(42))
            # Concatinate all
            train_datasets = [train_dataset_df, train_dataset_h36]
            test_datasets = [test_dataset_df, test_dataset_h36]
            self.train_dataset = ConcatDataset(train_datasets)
            self.test_dataset = ConcatDataset(test_datasets)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.arg.batch_size, shuffle=True, num_workers=4)
        return train_loader

    def val_dataloader(self):
        test_loader = DataLoader(self.test_dataset, batch_size=self.arg.batch_size, shuffle=True, num_workers=4)
        return test_loader

def main(arg):
    model = PLModel(arg)
    wandb_logger = WandbLogger(project="Bachelorarbeit", name=arg.name)
    # wandb_logger.watch(model)
    # wandb_logger.log_hyperparams(arg)
    trainer = pl.Trainer(gpus=arg.gpu, logger=wandb_logger, accelerator='ddp_spawn')
    trainer.fit(model)

if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)