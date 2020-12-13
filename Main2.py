import torch
from DataLoader import ImageDataset, DataLoader
from utils import save_model, load_model, load_deep_fashion_dataset, make_visualization
from Model2 import Model2
from config import parse_args, write_hyperparameters
from dotmap import DotMap
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
    bn = arg.batch_size
    mode = arg.mode
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    lr = arg.lr
    L_inv_scal = arg.L_inv_scal
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
        model = Model2(arg).to(device)
        if load_from_ckpt:
            model = load_model(model, model_save_dir, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Log with wandb
        wandb.init(project='Disentanglement', config=arg, name=arg.name)
        wandb.watch(model, log='all')
        # Load Datasets and DataLoader
        train_data, test_data = load_deep_fashion_dataset()
        train_dataset = ImageDataset(train_data)
        test_dataset = ImageDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=bn, num_workers=4)

        # Make Training
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(epochs+1):
                # Train on Train Set
                model.train()
                model.mode = 'train'
                for step, original in enumerate(train_loader):
                    original = original.to(device)
                    image_rec, reconstruct_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv = model(original)
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
                        transform_loss_log = torch.tensor([transform_loss])
                        precision_loss_log = torch.tensor([precision_loss])
                    else:
                        loss_log = torch.cat([loss_log, torch.tensor([loss])])
                        rec_loss_log = torch.cat([rec_loss_log, torch.tensor([rec_loss])])
                        transform_loss_log = torch.cat([transform_loss_log, torch.tensor([transform_loss])])
                        precision_loss_log = torch.cat([precision_loss_log, torch.tensor([precision_loss])])
                    training_loss = torch.mean(loss_log)
                    training_rec_loss = torch.mean(rec_loss_log)
                    training_transform_loss = torch.mean(transform_loss_log)
                    training_precision_loss = torch.mean(precision_loss_log)
                    wandb.log({"Training Loss": training_loss})
                    wandb.log({"Training Rec Loss": training_rec_loss})
                    wandb.log({"Training Transform Loss": training_transform_loss})
                    wandb.log({"Training Precision Loss": training_precision_loss})
                #print(f'Epoch: {epoch}, Train Loss: {training_loss}')

                # Evaluate on Test Set
                model.eval()
                for step, original in enumerate(test_loader):
                    with torch.no_grad():
                        original = original.to(device)
                        image_rec, reconstruct_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv = model(original)
                        if step == 0:
                            loss_log = torch.tensor([loss])
                        else:
                            loss_log = torch.cat([loss_log, torch.tensor([loss])])
                evaluation_loss = torch.mean(loss_log)
                wandb.log({"Evaluation Loss": evaluation_loss})
                #print(f'Epoch: {epoch}, Test Loss: {evaluation_loss}')

                # Track Progress & Visualization
                for step, original in enumerate(test_loader):
                    with torch.no_grad():
                        model.mode = 'predict'
                        original = original.to(device)
                        original_part_maps, image_rec, part_maps, part_maps, reconstruct_same_id = model(original)
                        make_visualization(original, original_part_maps, reconstruct_same_id, image_rec[:original.shape[0]],
                                           image_rec[original.shape[0]:], part_maps[original.shape[0]:],
                                           part_maps[:original.shape[0]], L_inv_scal, model_save_dir + '/summary/', epoch, device)
                        save_model(model, model_save_dir)

                        if step == 0:
                            break

    elif mode == 'predict':
        # Make Directory for Predictions
        model_save_dir = '../results/' + name
        prediction_save_dir = model_save_dir + '/predictions/'
        if not os.path.exists(prediction_save_dir):
            os.makedirs(prediction_save_dir)

        # Load Model and Dataset
        model = Model2(arg).to(device)
        model = load_model(model, model_save_dir, device)
        train_data, test_data = load_deep_fashion_dataset()
        test_dataset = ImageDataset(np.array(test_data))
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=bn, num_workers=4)
        model.mode = 'predict'
        model.eval()

        # Predict on Dataset
        for step, original in enumerate(test_loader):
            with torch.no_grad():
                original = original.to(device)
                original_part_maps, image_rec, part_maps, part_maps, reconstruct_same_id = model(original)
                make_visualization(original, original_part_maps, reconstruct_same_id, image_rec[:original.shape[0]],
                                   image_rec[original.shape[0]:], part_maps[original.shape[0]:],
                                   part_maps[:original.shape[0]], L_inv_scal, prediction_save_dir, 0, device)
                if step == 0:
                    break


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)

