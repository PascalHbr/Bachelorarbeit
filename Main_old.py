import torch
from Dataloader import DataLoader, get_dataset
from utils import save_model, load_model, visualize_predictions, keypoint_metric
from Model_old import Model
from config import parse_args, write_hyperparameters
from dotmap import DotMap
import os
import numpy as np
import wandb
from utils import count_parameters


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
    L_inv_scal = arg.L_inv_scal
    epochs = arg.epochs
    device = torch.device('cuda:' + str(arg.gpu[0]) if torch.cuda.is_available() else 'cpu')
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
    test_loader = DataLoader(test_dataset, batch_size=bn, shuffle=False, num_workers=4)

    if mode == 'train':
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
                    if epoch != 0:
                        model.L_sep = 0.
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
                        original_part_maps, mu_original, image_rec, part_maps, part_maps, reconstruction = model(original)
                        # img = visualize_predictions(original, original_part_maps, keypoints, reconstruction, image_rec[:original.shape[0]],
                        #                    image_rec[original.shape[0]:], part_maps[original.shape[0]:], part_maps[:original.shape[0]],
                        # #                    L_inv_scal, model_save_dir + '/summary/', epoch, device, show_labels=False)
                        # if epoch % 5 == 0:
                        #     wandb.log({"Summary_" + str(epoch): [wandb.Image(img)]})
                        save_model(model, model_save_dir)

                        if step == 0:
                            break
                # Decrements
                # model.L_sep = arg.sig_decr * model.L_sep

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
                ground_truth_images, img_reconstr, mu, L_inv, part_map_norm, heat_map, heat_map_norm, total_loss = model(
                    original)
                score, mu_new, L_inv, part_map_norm_new, heat_map_new = keypoint_metric(mu, keypoints, L_inv,
                                                                                        part_map_norm, heat_map,
                                                                                        arg.reconstr_dim)
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

