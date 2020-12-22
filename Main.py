import torch
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, make_visualization, keypoint_metric
from Model import Model
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from utils import count_parameters


def main(arg):
    # Set random seeds
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(7)
    rng = np.random.RandomState(7)

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

    # Choose Dataset
    dataset = get_dataset(arg.dataset)

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
        print(f'Number of Parameters: {count_parameters(model)}')
        if load_from_ckpt:
            model = load_model(model, model_save_dir, device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Log with wandb
        wandb.init(project='Disentanglement', config=arg, name=arg.name)
        wandb.watch(model, log='all')
        # Load Datasets and DataLoader
        train_dataset = dataset(size=arg.reconstr_dim, train=True)
        test_dataset = dataset(size=arg.reconstr_dim, train=False)
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=bn, num_workers=4)

        # Make Training
        with torch.autograd.set_detect_anomaly(False):
            for epoch in range(epochs+1):
                # Train on Train Set
                model.train()
                model.mode = 'train'
                for step, (original, keypoints) in enumerate(train_loader):
                    original, keypoints = original.to(device), keypoints.to(device)
                    image_rec, reconstruct_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv, mu_original = model(original)
                    mu_norm = torch.mean(torch.norm(mu, p=1, dim=2)).cpu().detach().numpy()
                    L_inv_norm = torch.mean(torch.linalg.norm(L_inv, ord='fro', dim=[2, 3])).cpu().detach().numpy()
                    # Track Mean and Precision Matrix
                    wandb.log({"Part Means": mu_norm})
                    wandb.log({"Precision Matrix": L_inv_norm})
                    # Zero out gradients
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), arg.clip)
                    optimizer.step()
                    # Track Loss
                    wandb.log({"Training Loss": loss})
                    # Track Metric
                    score = keypoint_metric(mu_original, keypoints)
                    wandb.log({"Metric Train": score})

                # Evaluate on Test Set
                model.eval()
                val_score = torch.zeros(1)
                val_loss = torch.zeros(1)
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        original, keypoints = original.to(device), keypoints.to(device)
                        image_rec, reconstruct_same_id, loss, rec_loss, transform_loss, precision_loss, mu, L_inv, mu_original = model(original)
                        # Track Loss and Metric
                        score = keypoint_metric(mu_original, keypoints)
                        val_score += score.cpu()
                        val_loss += loss.cpu()

                val_loss = val_loss / (step + 1)
                val_score = val_score / (step + 1)
                wandb.log({"Evaluation Loss": val_loss})
                wandb.log({"Metric Validation": val_score})

                # Track Progress & Visualization
                for step, (original, keypoints) in enumerate(test_loader):
                    with torch.no_grad():
                        model.mode = 'predict'
                        original, keypoints = original.to(device), keypoints.to(device)
                        original_part_maps, mu_original, image_rec, part_maps, part_maps, reconstruct_same_id = model(original)
                        img = make_visualization(original, original_part_maps, keypoints, reconstruct_same_id, image_rec[:original.shape[0]],
                                           image_rec[original.shape[0]:], part_maps[original.shape[0]:], part_maps[:original.shape[0]],
                                           L_inv_scal, model_save_dir + '/summary/', epoch, device, show_labels=False)
                        if epoch % 5 == 0:
                            wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        save_model(model, model_save_dir)

                        if step == 0:
                            break
                # Decrements
                # model.L_sep = arg.sig_decr * model.L_sep

    elif mode == 'predict':
        # Make Directory for Predictions
        model_save_dir = '../results/' + name
        prediction_save_dir = model_save_dir + '/predictions/'
        if not os.path.exists(prediction_save_dir):
            os.makedirs(prediction_save_dir)

        # Load Model and Dataset
        model = Model(arg).to(device)
        model = load_model(model, model_save_dir, device)
        test_dataset = dataset(size=arg.reconstr_dim, train=False)
        test_loader = DataLoader(test_dataset, batch_size=bn, num_workers=4)
        model.mode = 'predict'
        model.eval()

        # Predict on Dataset
        for step, (original, keypoints) in enumerate(test_loader):
            with torch.no_grad():
                original, keypoints = original.to(device), keypoints.to(device)
                original_part_maps, mu_original, image_rec, part_maps, part_maps, reconstruct_same_id = model(original)
                make_visualization(original, original_part_maps, keypoints, reconstruct_same_id, image_rec[:original.shape[0]],
                                   image_rec[original.shape[0]:], part_maps[original.shape[0]:], part_maps[:original.shape[0]],
                                   L_inv_scal, prediction_save_dir, 0, device, show_labels=True)
                # Track Metric
                score = keypoint_metric(mu_original, keypoints)
                if step == 0:
                    break


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)

