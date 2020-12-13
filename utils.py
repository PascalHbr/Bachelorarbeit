import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import csv
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from architecture import softmax
from ops import get_heat_map, get_mu_and_prec
import cv2


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir, device):
    model.load_state_dict(torch.load(model_save_dir + '/parameters', map_location=device))
    return model


def load_images_from_folder():
    folder = "/export/scratch/compvis/datasets/deepfashion_vunet/train"
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_deep_fashion_dataset(stage=None):
    data_folder = "/export/scratch/compvis/datasets/deepfashion_inshop/Img/img/"
    csv_folder = "/export/scratch/compvis/datasets/compvis-datasets/deepfashion_allJointsVisible/"
    train_images = []
    test_images = []

    if stage == 'fit' or stage is None:

        with open(csv_folder + "data_train.csv") as train_file:
            train_reader = csv.reader(train_file)
            next(train_reader)  # ignore first row
            for row in train_reader:
                img_path = row[1]
                img = plt.imread(os.path.join(data_folder, img_path))
                train_images.append(img)

        with open(csv_folder + "data_test.csv") as test_file:
            test_reader = csv.reader(test_file)
            next(test_reader)  # ignore first row
            for row in test_reader:
                img_path = row[1]
                img = plt.imread(os.path.join(data_folder, img_path))
                test_images.append(img)

        return np.array(train_images), np.array(test_images)

    if stage == 'test':

        with open(csv_folder + "data_test.csv") as test_file:
            test_reader = csv.reader(test_file)
            next(test_reader)  # ignore first row
            for row in test_reader:
                img_path = row[1]
                img = plt.imread(os.path.join(data_folder, img_path))
                test_images.append(img)

        return np.array(test_images)


def make_visualization(original, original_part_maps, reconstruction, shape_transform, app_transform, fmap_shape,
                       fmap_app, L_inv_scale, directory, epoch, device, index=0):

    # Color List for Parts
    color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine',
                  'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen']
    # Get Maps
    fmap_shape_norm = softmax(fmap_shape)
    mu_shape, L_inv_shape = get_mu_and_prec(fmap_shape_norm, device, L_inv_scale)
    heat_map_shape = get_heat_map(mu_shape, L_inv_shape, device)

    fmap_app_norm = softmax(fmap_app)
    mu_app, L_inv_app = get_mu_and_prec(fmap_app_norm, device, L_inv_scale)
    heat_map_app = get_heat_map(mu_app, L_inv_app, device)

    overlay_original, img_with_marker = visualize_keypoints(original, original_part_maps, L_inv_scale, device)
    cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                    ['white', color_list[0]],
                                                    256)

    with PdfPages(directory + str(epoch) + '_summary.pdf') as pdf:
        # Make Head with Overview
        fig_head, axs_head = plt.subplots(6, 4, figsize=(12, 12))
        fig_head.suptitle("Overview", fontsize="x-large")
        axs_head[0, 0].imshow(original[index].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[0, 1].imshow(app_transform[index].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[0, 2].imshow(shape_transform[index].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[0, 3].imshow(reconstruction[index].permute(1, 2, 0).cpu().detach().numpy())

        axs_head[1, 0].imshow(app_transform[index].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[1, 2].imshow(shape_transform[index].permute(1, 2, 0).cpu().detach().numpy())

        axs_head[2, 0].imshow(reconstruction[0].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[2, 1].imshow(reconstruction[1].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[2, 2].imshow(reconstruction[2].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[2, 3].imshow(reconstruction[3].permute(1, 2, 0).cpu().detach().numpy())

        axs_head[3, 0].imshow(original[0].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[3, 1].imshow(original[1].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[3, 2].imshow(original[2].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[3, 3].imshow(original[3].permute(1, 2, 0).cpu().detach().numpy())

        axs_head[4, 0].imshow(overlay_original[0], cmap=cmap)
        axs_head[4, 1].imshow(overlay_original[1], cmap=cmap)
        axs_head[4, 2].imshow(overlay_original[2], cmap=cmap)
        axs_head[4, 3].imshow(overlay_original[3], cmap=cmap)

        axs_head[5, 0].imshow(img_with_marker[0])
        axs_head[5, 1].imshow(img_with_marker[1])
        axs_head[5, 2].imshow(img_with_marker[2])
        axs_head[5, 3].imshow(img_with_marker[3])

        # Part Visualization Shape Stream
        fig_shape, axs_shape = plt.subplots(8, 6, figsize=(8, 8))
        fig_shape.suptitle("Part Visualization Shape Stream", fontsize="x-large")
        for i in range(16):
            cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                            ['white', color_list[i]],
                                                            256)
            if i == 0:
                overlay_shape = heat_map_shape[index][i]
            else:
                overlay_shape += heat_map_shape[index][i]

            axs_shape[int(i / 2), (i % 2) * 3].imshow(fmap_shape[index][i].cpu().detach().numpy(), cmap=cmap)
            axs_shape[int(i / 2), (i % 2) * 3 + 1].imshow(fmap_shape_norm[index][i].cpu().detach().numpy(), cmap=cmap)
            axs_shape[int(i / 2), (i % 2) * 3 + 2].imshow(heat_map_shape[index][i].cpu().detach().numpy(), cmap=cmap)

            if i == 15:
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['white', 'black'],
                                                                256)
                axs_head[1, 1].imshow(overlay_shape.cpu().detach().numpy(), cmap=cmap)

        # Part Visualization Appearance Stream
        fig_app, axs_app = plt.subplots(8, 6, figsize=(8, 8))
        fig_app.suptitle("Part Visualization Appearance Stream", fontsize="x-large")
        for i in range(16):
            cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                            ['white', color_list[i]],
                                                            256)
            if i == 0:
                overlay_app = heat_map_app[index][i]
            else:
                overlay_app += heat_map_app[index][i]

            axs_app[int(i / 2), (i % 2) * 3].imshow(fmap_app[index][i].cpu().detach().numpy(), cmap=cmap)
            axs_app[int(i / 2), (i % 2) * 3 + 1].imshow(fmap_app_norm[index][i].cpu().detach().numpy(), cmap=cmap)
            axs_app[int(i / 2), (i % 2) * 3 + 2].imshow(heat_map_app[index][i].cpu().detach().numpy(), cmap=cmap)

            if i == 15:
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['white', 'black'],
                                                                256)
                axs_head[1, 3].imshow(overlay_app.cpu().detach().numpy(), cmap=cmap)

        pdf.savefig(fig_head)
        pdf.savefig(fig_shape)
        pdf.savefig(fig_app)

        plt.close('all')


def visualize_keypoints(img, fmap, L_inv_scale, device):
    # Make Heatmap Overlay
    fmap_norm = softmax(fmap)
    mu, L_inv = get_mu_and_prec(fmap_norm, device, L_inv_scale)
    heat_map = get_heat_map(mu, L_inv, device)
    heat_map_overlay = torch.sum(heat_map, dim=1).cpu().detach().numpy()

    # Mark Keypoints
    img, mu = img.permute(0, 2, 3, 1).cpu().detach().numpy(), mu.cpu().detach().numpy()
    img = np.ascontiguousarray(img)
    mu_scale = ((mu + 1.) / 2. * img.shape[1])
    n_parts = mu.shape[1]
    for i, image in enumerate(img):
        for k in range(n_parts):
            cv2.drawMarker(image, (mu_scale[i][k][1], mu_scale[i][k][0]), (1.,0,0), markerType=cv2.MARKER_CROSS,
                           markerSize=15, thickness=1, line_type=cv2.LINE_AA)

    return heat_map_overlay, img
