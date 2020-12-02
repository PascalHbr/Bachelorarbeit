import torch
from DataLoader import ImageDataset, ImageDataset2, DataLoader
from utils import save_model, load_model, load_images_from_folder, make_visualization
from Model import Model
from config import parse_args, write_hyperparameters
from dotmap import DotMap
from ops import normalize
import os
import numpy as np
from torchvision.utils import save_image
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline
import kornia.augmentation as K


def main(arg):
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
        if load_from_ckpt == True:
            model = load_model(model, model_save_dir).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Load Datasets and DataLoader
        data = load_images_from_folder()
        train_data = np.array(data[:-10])
        train_dataset = ImageDataset(train_data)
        test_data = np.array(data[-10:])
        test_dataset = ImageDataset(test_data)
        train_loader = DataLoader(train_dataset, batch_size=bn, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=bn, num_workers=4)

        # Make Training
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
                # Zero out gradients
                optimizer.zero_grad()
                prediction, loss = model(original, image_spatial_t, image_appearance_t, coord, vector)
                loss.backward()
                optimizer.step()
                if step == 0:
                    loss_log = torch.tensor([loss])
                else:
                    loss_log = torch.cat([loss_log, torch.tensor([loss])])
            print(f'Epoch: {epoch}, Train Loss: {torch.mean(loss_log)}')

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
                    prediction, loss = model(original, image_spatial_t, image_appearance_t, coord, vector)
                    if step == 0:
                        loss_log = torch.tensor([loss])
                    else:
                        loss_log = torch.cat([loss_log, torch.tensor([loss])])
            print(f'Epoch: {epoch}, Test Loss: {torch.mean(loss)}')

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
        data = load_images_from_folder()
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
                save_image(image[0], model_save_dir + '/predictions/original.png')
                save_image(reconstruction[0], model_save_dir + '/predictions/reconstruction.png')
                save_image(image_spatial_t[0], model_save_dir + '/predictions/spat0.png')
                save_image(image_spatial_t[1], model_save_dir + '/predictions/spat1.png')
                save_image(image_spatial_t[2], model_save_dir + '/predictions/spat2.png')
                save_image(image_spatial_t[3], model_save_dir + '/predictions/spat3.png')


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)

