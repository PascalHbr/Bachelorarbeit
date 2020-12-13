import torch
from DataLoader import ImageDataset, DataLoader
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
