import torch
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, keypoint_metric, visualize_results, count_parameters, visualize_predictions
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from torch.utils.data import ConcatDataset, random_split
from Model import Model
from torch.optim.lr_scheduler import  ReduceLROnPlateau


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
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')
    arg.device = device

    # Load Datasets and DataLoader
    if arg.dataset != "mix":
        dataset = get_dataset(arg.dataset)
    if arg.dataset == 'pennaction':
        # init_dataset = dataset(size=arg.reconstr_dim, action_req=["tennis_serve", "tennis_forehand", "baseball_pitch",
        #                                                           "baseball_swing", "jumping_jacks", "golf_swing"])
        init_dataset = dataset(size=arg.reconstr_dim)
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
        model_save_dir = '../results/' + arg.dataset + '/' + name
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/summary')

        # Save Hyperparameters
        write_hyperparameters(arg.toDict(), model_save_dir)

        # Define Model
        model = Model(arg)
        if len(arg.gpu) > 1:
            model = torch.nn.DataParallel(model, device_ids=arg.gpu)
        model.to(device)
        if load_from_ckpt:
            model = load_model(model, model_save_dir, device).to(device)
        # Dataparallel
        print(arg.gpu)
        print(f'Number of Parameters: {count_parameters(model)}')

        # Definde Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=arg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, threshold=1e-4, patience=6)

        # Log with wandb
        wandb.init(project='Disentanglement', config=arg, name=arg.name)
        wandb.watch(model, log='all')

        # Make Training
        with torch.autograd.set_detect_anomaly(False):
            for epoch in range(epochs+1):
                # Train on Train Set
                model.train()
                # model.mode = 'train'
                for step, (original, keypoints) in enumerate(train_loader):
                    bn = original.shape[0]
                    original, keypoints = original.to(device), keypoints.to(device)
                    # Forward Pass
                    ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, total_loss = model(original)
                    # Track Mean and Precision Matrix
                    mu_norm = torch.mean(torch.norm(mu[:bn], p=1, dim=2)).cpu().detach().numpy()
                    L_inv_norm = torch.mean(torch.linalg.norm(L_inv[:bn], ord='fro', dim=[2, 3])).cpu().detach().numpy()
                    wandb.log({"Part Means": mu_norm})
                    wandb.log({"Precision Matrix": L_inv_norm})
                    # Zero out gradients
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()
                    # Track Loss
                    wandb.log({"Training Loss": total_loss.cpu()})
                    # Track Metric
                    score, mu, L_inv, part_map_norm, heat_map = keypoint_metric(mu, keypoints, L_inv,
                                                                         part_map_norm, heat_map, arg.reconstr_dim)
                    wandb.log({"Metric Train": score})
                    # Track progress
                    if step % 10000 == 0 and bn >= 4:
                        for step_, (original, keypoints) in enumerate(test_loader):
                            with torch.no_grad():
                                original, keypoints = original.to(device), keypoints.to(device)
                                ground_truth_images, img_reconstr, mu, L_inv, part_map_norm,\
                                heat_map, heat_map_norm, total_loss = model(original)
                                # Visualize Results
                                score, mu, L_inv, part_map_norm, heat_map = keypoint_metric(mu, keypoints, L_inv,
                                                                                            part_map_norm, heat_map, arg.reconstr_dim)
                                img = visualize_results(ground_truth_images, img_reconstr, mu, L_inv, part_map_norm,
                                                    heat_map, keypoints, model_save_dir + '/summary/', epoch, arg.background)
                                wandb.log({"Summary at step" + str(step): [wandb.Image(img)]})
                                save_model(model, model_save_dir)
                                if step_ == 0:
                                    break


                # Evaluate on Test Set
                model.eval()
                val_score = torch.zeros(1)
                val_loss = torch.zeros(1)
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        original, keypoints = original.to(device), keypoints.to(device)
                        ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, total_loss= model(original)
                        # Track Loss and Metric
                        score, mu, L_inv, part_map_norm, heat_map = keypoint_metric(mu, keypoints, L_inv,
                                                                             part_map_norm, heat_map, arg.reconstr_dim)
                        val_score += score.cpu()
                        val_loss += total_loss.cpu()

                val_loss = val_loss / (step + 1)
                val_score = val_score / (step + 1)
                if epoch == 0:
                    best_score = val_score
                if val_score <= best_score:
                    best_score = val_score
                save_model(model, model_save_dir)
                scheduler.step(val_score)
                wandb.log({"Evaluation Loss": val_loss})
                wandb.log({"Metric Validation": val_score})

                # Track Progress & Visualization
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        original, keypoints = original.to(device), keypoints.to(device)
                        ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, total_loss = model(original)
                        score, mu, L_inv, part_map_norm, heat_map = keypoint_metric(mu, keypoints, L_inv,
                                                                                    part_map_norm, heat_map, arg.reconstr_dim)
                        img = visualize_results(ground_truth_images, img_reconstr, mu, L_inv, part_map_norm,
                                                heat_map, keypoints, model_save_dir + '/summary/', epoch, arg.background)
                        wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        if step == 0:
                            break

    elif mode == 'predict':
        # Make Directory for Predictions
        model_save_dir = '../results/' + arg.dataset + '/' + name
        # Dont use Transformations
        arg.tps_scal = 0.
        arg.rot_scal = 0.
        arg.off_scal = 0.
        arg.scal_var = 0.
        arg.augm_scal = 1.
        arg.contrast = 0.
        arg.brightness = 0.
        arg.saturation = 0.
        arg.hue = 0.

        # Load Model and Dataset
        model = Model(arg).to(device)
        model = load_model(model, model_save_dir, device)
        model.eval()

        # Log with wandb
        # wandb.init(project='Disentanglement', config=arg, name=arg.name)
        # wandb.watch(model, log='all')

        # Predict on Dataset
        val_score = torch.zeros(1)
        for step, (original, keypoints) in enumerate(test_loader):
            with torch.no_grad():
                original, keypoints = original.to(device), keypoints.to(device)
                ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, total_loss = model(original)
                score, mu_new, L_inv, part_map_norm_new, heat_map_new = keypoint_metric(mu, keypoints, L_inv,
                                                                            part_map_norm, heat_map, arg.reconstr_dim)
                if step == 0:
                    img = visualize_predictions(original, img_reconstr, mu_new, part_map_norm_new, heat_map_new, mu,
                                            part_map_norm, heat_map, model_save_dir)
                # wandb.log({"Prediction": [wandb.Image(img)]})
                val_score += score.cpu()

        val_score = val_score / (step + 1)
        print("Validation Score: ", val_score)


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)
