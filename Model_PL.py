import torch
import torch.nn.functional as F
from DataLoader import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from opt_einsum import contract
from architecture_ops import E, Decoder
from ops_old import feat_mu_to_enc, get_local_part_appearances, get_mu_and_prec
from ops import prepare_pairs, AbsDetJacobian, loss_fn
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
from utils import load_deep_fashion_dataset
from config import parse_args
from dotmap import DotMap
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")
    def on_train_end(self, trainer, pl_module):
        print("Training is done.")

class DeepFashionDataModule(pl.LightningDataModule):

    def __init__(self, arg):
        super().__init__()
        self.transform = transforms.ToTensor()
        self.batch_size = arg.batch_size

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_data, self.val_data = load_deep_fashion_dataset(stage)
            self.train_dataset, self.val_dataset = ImageDataset(self.train_data), ImageDataset(self.val_data)

        if stage == 'test':
            self.test_data = load_deep_fashion_dataset(stage)
            self.test_dataset = ImageDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)


class Model(pl.LightningModule):

    def __init__(self, arg):
        super().__init__()
        self.arg = arg
        self.mode = arg.mode
        self.batch_size = arg.batch_size
        self.lr = arg.lr
        self.reconstr_dim = arg.reconstr_dim
        self.n_parts = arg.n_parts
        self.n_features = arg.n_features
        #self.device = arg.device
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
        self.rot_scal = arg.rot_scal
        self.off_scal = arg.off_scal
        self.scal_var = arg.scal_var
        self.augm_scal = arg.augm_scal
        self.fold_with_shape = arg.fold_with_shape
        self.E_sigma = E(self.depth_s, self.n_parts, self.residual_dim, self.p_dropout, sigma=True)
        self.E_alpha = E(self.depth_a, self.n_features, self.residual_dim, self.p_dropout, sigma=False)
        self.decoder = Decoder(self.n_parts, self.n_features, self.reconstr_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        batch_size2 = 2 * x.shape[0]
        # tps
        image_orig = x.repeat(2, 1, 1, 1)
        tps_param_dic = tps_parameters(batch_size2, self.scal, self.tps_scal, self.rot_scal, self.off_scal,
                                       self.scal_var, self.augm_scal)
        coord, vector = make_input_tps_param(tps_param_dic)
        coord, vector = coord.to(self.device), vector.to(self.device)
        t_images, t_mesh = ThinPlateSpline(image_orig, coord, vector, self.reconstr_dim, device=self.device)
        image_in, image_rec = prepare_pairs(t_images, self.arg, self.device)
        transform_mesh = F.interpolate(t_mesh, size=64)
        volume_mesh = AbsDetJacobian(transform_mesh, self.device)

        # encoding
        part_maps_raw, part_maps_norm, sum_part_maps = self.E_sigma(image_in)
        mu, L_inv = get_mu_and_prec(part_maps_norm, self.device, self.L_inv_scal)
        raw_features = self.E_alpha(sum_part_maps)
        features = get_local_part_appearances(raw_features, part_maps_norm)

        # transform
        integrant = (part_maps_norm.unsqueeze(-1) * volume_mesh.unsqueeze(-1)).squeeze()
        integrant = integrant / torch.sum(integrant, dim=[2, 3], keepdim=True)
        mu_t = contract('akij, alij -> akl', integrant, transform_mesh)
        transform_mesh_out_prod = contract('amij, anij -> amnij', transform_mesh, transform_mesh)
        mu_out_prod = contract('akm, akn -> akmn', mu_t, mu_t)
        stddev_t = contract('akij, amnij -> akmn', integrant, transform_mesh_out_prod) - mu_out_prod

        # processing
        encoding = feat_mu_to_enc(features, mu, L_inv, self.device, self.covariance, self.reconstr_dim)
        reconstruct_same_id = self.decoder(encoding)

        total_loss, rec_loss, transform_loss, precision_loss = loss_fn(batch_size, mu, L_inv, mu_t, stddev_t,
                                                                       reconstruct_same_id, image_rec, self.fold_with_shape,
                                                                       self.l_2_scal, self.l_2_threshold, self.L_mu, self.L_cov,
                                                                       self.device)

        if self.mode == 'predict':
            original_part_maps_raw, original_part_maps_norm, original_sum_part_maps = self.E_sigma(x)
            return original_part_maps_raw, image_rec, part_maps_raw, part_maps_raw, reconstruct_same_id

        elif self.mode == 'train':
            return image_rec, reconstruct_same_id, total_loss, rec_loss, transform_loss, precision_loss, mu, L_inv

    def training_step(self, batch, batch_idx):
        image_rec, reconstruct_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv = self(batch)
        self.log('my_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image_rec, reconstruct_same_id, val_loss, rec_loss, transform_loss, precision_loss, mu, L_inv = self(batch)
        self.log('my_loss', val_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        image_rec, reconstruct_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv = self(batch)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


def main(arg):
    seed_everything(42)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [PrintCallback(), lr_monitor]
    logger = WandbLogger(name=arg.name, save_dir='../results/' + arg.name, )
    trainer = Trainer(gpus=arg.gpu,
                      deterministic=True,
                      benchmark=False,
                      callbacks=callbacks,
                      checkpoint_callback=False,
                      logger=logger,
                      max_epochs=1000,
                      precision=32,
                      weights_save_path='../results/' + arg.name,
                      weights_summary=None
                      )
    dm = DeepFashionDataModule(arg)
    model = Model(arg)
    trainer.tune(model, dm)
    trainer.fit(model, dm)


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)