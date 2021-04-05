import torch
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, keypoint_metric, visualize_SAE, count_parameters, visualize_predictions
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from torch.utils.data import ConcatDataset, random_split
from Model import Model


def coordinate_transformation(coords, grid, device, grid_size=1000):
    bn, k, _ = coords.shape
    bucket = torch.linspace(-10., 10., grid_size, device=device)
    indices = torch.bucketize(coords.contiguous(), bucket)
    indices = indices.unsqueeze(-2).unsqueeze(-2)
    grid = grid.unsqueeze(1).repeat(1, k, 1, 1, 1)
    new_coords = torch.gather(grid, 3, indices).squeeze(-2).squeeze(-2)

    return new_coords


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

    if arg.mode == 'train':
        # Make new directory
        model_save_dir = '../results/' + name
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
            os.makedirs(model_save_dir + '/summary')

        # Save Hyperparameters
        write_hyperparameters(arg.toDict(), model_save_dir)

        # Define Model
        model = Model(arg).to(device)
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
                # model.mode = 'train'
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
                    # Track progress
                    if step % 10000 == 0 and bn >= 4:
                        for step_, (original, keypoints) in enumerate(test_loader):
                            with torch.no_grad():
                                original, keypoints = original.to(device), keypoints.to(device)
                                ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm, total_loss = model(
                                    original)

                                img = visualize_SAE(ground_truth_images, img_reconstr, mu, prec, part_map_norm,
                                                    heat_map_norm,
                                                    keypoints, model_save_dir + '/summary/', epoch)
                                wandb.log({"Summary at step" + str(step): [wandb.Image(img)]})
                                if step_ == 0:
                                    break


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
                        original, keypoints = original.to(device), keypoints.to(device)
                        ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm, total_loss = model(original)

                        img = visualize_SAE(ground_truth_images, img_reconstr, mu, prec, part_map_norm, heat_map_norm,
                                            keypoints, model_save_dir + '/summary/', epoch)
                        wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        save_model(model, model_save_dir)

                        if step == 0:
                            break

    elif arg.mode == 'predict':
        # Make Directory for Predictions
        model_save_dir = '../results/' + name
        prediction_save_dir = model_save_dir + '/predictions/'
        if not os.path.exists(prediction_save_dir):
            os.makedirs(prediction_save_dir)

        # Load Model and Dataset
        model = Model(arg).to(device)
        model = load_model(model, model_save_dir, device)
        model.eval()

        # Log with wandb
        wandb.init(project='Disentanglement', config=arg, name=arg.name)
        wandb.watch(model, log='all')

        # Predict on Dataset
        for step, (original, keypoints) in enumerate(test_loader):
            with torch.no_grad():
                original, keypoints = original.to(device), keypoints.to(device)
                mu = model(original)
                img = visualize_predictions(original, mu, model_save_dir)
                wandb.log({"Prediction": [wandb.Image(img)]})

                if step == 0:
                    break


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)
