import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from architecture import softmax
from ops import get_heat_map, get_mu_and_prec

import os
from glob import glob
import cv2
from natsort import natsorted


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir, device):
    model.load_state_dict(torch.load(model_save_dir + '/parameters', map_location=device))
    return model


def make_visualization(original, original_part_maps, labels, reconstruction, shape_transform, app_transform, fmap_shape,
                       fmap_app, L_inv_scale, directory, epoch, device, index=0, show_labels=False):

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

    overlay_original, img_with_marker = visualize_keypoints(original, original_part_maps, labels, L_inv_scale, device,
                                                            show_labels)
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


def visualize_keypoints(img, fmap, labels, L_inv_scale, device, show_labels):
    # Make Heatmap Overlay
    fmap_norm = softmax(fmap)
    mu, L_inv = get_mu_and_prec(fmap_norm, device, L_inv_scale)
    heat_map = get_heat_map(mu, L_inv, device)
    heat_map_overlay = torch.sum(heat_map, dim=1).cpu().detach().numpy()

    # Mark Keypoints
    img, mu = img.permute(0, 2, 3, 1).cpu().detach().numpy(), mu.cpu().detach().numpy()
    img = np.ascontiguousarray(img)
    mu_scale = (mu + 1.) / 2. * img.shape[1]
    labels = labels[:, 0].cpu().detach().numpy()
    n_parts = mu.shape[1]
    n_labels = labels.shape[1]
    for i, image in enumerate(img):
        for k in range(n_parts):
            cv2.drawMarker(image, (int(mu_scale[i][k][1]), int(mu_scale[i][k][0])), (1., 0, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1, line_type=cv2.LINE_AA)

        if show_labels:
            for n in range(n_labels):
                cv2.drawMarker(image, (int(labels[i][n][1]), int(labels[i][n][0])), (0, 1., 0),
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1, line_type=cv2.LINE_AA)

    return heat_map_overlay, img


def keypoint_metric(prediction, ground_truth, image_size=256):
    bn, nk, _ = prediction.shape
    prediction = ((prediction + 1.) / 2. * image_size).float().cpu()
    ground_truth = ground_truth[:, 0].float().cpu()
    distances = torch.zeros(1)
    for i in range(nk):
        best_distance = 1e7
        for j in range(nk):
            distance = torch.mean(torch.cdist(prediction[:, j], ground_truth[:, i], p=2.0))
            if distance < best_distance:
                best_distance = distance
        distances += best_distance
    distance_norm = distances / (nk * image_size)

    return distance_norm


if __name__ == '__main__':
    basepath = "/export/scratch/compvis/datasets/human3M_lorenz19"
    subdir_name = "test"
    datafiles = natsorted(glob(os.path.join(basepath, subdir_name, "*", "*", "*.jpg")))
    for i in range(50, 60):
        image = cv2.imread(datafiles[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print (datafiles[i])
        plt.imshow(image)
        plt.show()
