import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import cv2
from json import JSONEncoder
import matplotlib.colors as colors
from ops import get_heat_map
import json
from natsort import natsorted
import os
from glob import glob
import scipy.io


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir, device):
    model.load_state_dict(torch.load(model_save_dir + '/parameters', map_location=device))
    return model


def keypoint_metric(prediction, ground_truth, L_inv, part_map_norm, heat_map, image_size):
    bn, _, nk, _ = ground_truth.shape
    bi = L_inv.shape[0]
    preds = ((prediction + 1.) / 2. * image_size).float().detach().cpu()
    gt = ground_truth[:, 0].float().detach().cpu()
    first_indices = torch.arange(bn)[:, None]

    # sort by y index
    indices_gt = gt[:, :, 0].sort()[1]
    gt_sorted = gt[first_indices, indices_gt]

    # Make Empty Tensors
    keypoints_to_use = torch.randn([bi, nk, 2])
    part_map_norm_to_use = torch.randn([bi, nk, 64, 64])
    heat_map_to_use = torch.randn([bi, nk, 64, 64])
    L_inv_to_use = torch.randn([bi, nk, 2, 2])

    # Get Distances
    distances = torch.zeros(1)
    for b in range(bn):
        best_indices = []
        preds_bn = preds[b]
        gt_bn = gt_sorted[b]
        for k in range(nk):
            best_distance = 1e7
            best_index = 1e7
            for i in range(len(preds_bn)):
                distance = torch.mean(torch.cdist(preds_bn[i].unsqueeze(0), gt_bn[k].unsqueeze(0), p=2.0))
                if distance < best_distance and i not in best_indices:
                    best_distance = distance
                    best_index = i
            best_indices.append(best_index)
            keypoints_to_use[b, k, :] = prediction[b, best_index, :]
            part_map_norm_to_use[b, k, :, :] = part_map_norm[b, best_index, :, :]
            heat_map_to_use[b, k, :, :] = heat_map[b, best_index, :, :]
            L_inv_to_use[b, k, :] = L_inv[b, best_index, :]
            if bi == 2*bn:
                keypoints_to_use[b+bn, k, :] = prediction[b+bn, best_index, :]
                part_map_norm_to_use[b+bn, k, :, :] = part_map_norm[b+bn, best_index, :, :]
                heat_map_to_use[b+bn, k, :, :] = heat_map[b+bn, best_index, :, :]
                L_inv_to_use[b+bn, k, :] = L_inv[b+bn, best_index, :]
            distances += best_distance

    distance_norm = distances / (bn * nk * image_size)

    return distance_norm, keypoints_to_use, L_inv_to_use, part_map_norm_to_use, heat_map_to_use


def visualize_results(org, img_reconstr, mu, prec, part_map_norm,
                  heat_map, labels, directory, epoch, background, show_labels=True):
    bn = org.shape[0] // 2
    marker_size = 15 if org.shape[2] == 256 else 8

    # Neglect Background
    if background:
        part_map_norm = part_map_norm[:, :-1]
        heat_map = heat_map[:, :-1]

    # Make Maps
    part_map_overlay = torch.sum(part_map_norm, dim=1).cpu().detach().numpy()
    heat_map_overlay = torch.sum(heat_map, dim=1).cpu().detach().numpy()


    # Mark Keypoints
    original, mu = org.permute(0, 2, 3, 1).cpu().detach().numpy(), mu.cpu().detach().numpy()
    img = np.ascontiguousarray(original)
    mu_scale = (mu + 1.) / 2. * img.shape[1]
    labels = labels[:, 0].cpu().detach().numpy()
    n_parts = mu.shape[1]
    n_labels = labels.shape[1]
    for i, image in enumerate(img[:bn]):
        for k in range(n_parts):
            cv2.drawMarker(image, (int(mu_scale[i][k][1]), int(mu_scale[i][k][0])), (1., 0, 0),
                           markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=1, line_type=cv2.LINE_AA)

        if show_labels:
            for n in range(n_labels):
                cv2.drawMarker(image, (int(labels[i][n][1]), int(labels[i][n][0])), (0, 1., 0),
                               markerType=cv2.MARKER_CROSS, markerSize=marker_size, thickness=1, line_type=cv2.LINE_AA)

    with PdfPages(directory + str(epoch) + '_summary.pdf') as pdf:
        fig_head, axs_head = plt.subplots(4, 9, figsize=(15, 15))
        fig_head.suptitle("Overview", fontsize="x-large")
        for i in range(4):
            axs_head[i, 0].imshow(org[:bn][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 0].axis('off')
            axs_head[i, 1].imshow(1 - part_map_overlay[:bn][i], cmap='gray')
            axs_head[i, 1].axis('off')
            axs_head[i, 2].imshow(1 - heat_map_overlay[:bn][i], cmap='gray')
            axs_head[i, 2].axis('off')
            axs_head[i, 3].imshow(img_reconstr[:bn][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 3].axis('off')
            axs_head[i, 4].imshow(img[:bn][i])
            axs_head[i, 4].axis('off')
            axs_head[i, 5].imshow(org[bn:][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 5].axis('off')
            axs_head[i, 6].imshow(1 - part_map_overlay[bn:][i], cmap='gray')
            axs_head[i, 6].axis('off')
            axs_head[i, 7].imshow(1 - heat_map_overlay[bn:][i], cmap='gray')
            axs_head[i, 7].axis('off')
            axs_head[i, 8].imshow(img_reconstr[bn:][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 8].axis('off')

        pdf.savefig(fig_head)

        fig_head.canvas.draw()
        w, h = fig_head.canvas.get_width_height()
        img = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

        plt.close('all')

    return img


def visualize_predictions(org, reconstruction, mu, part_map, heat_map, mu_old, part_map_old, heat_map_old, directory, index=0):
    color_list = [(0,0,0), (255,255,255), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (174, 101, 0),
                  (192,192,192), (128,128,128), (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128), (0,0,128)]

    bn = org.shape[0]
    assert bn % 4 == 0
    original, mu = org.permute(0, 2, 3, 1).cpu().detach().numpy(), mu.cpu().detach().numpy()
    img = np.ascontiguousarray(original)
    mu_scale = (mu + 1.) / 2. * img.shape[1]
    n_parts = mu.shape[1]
    n_parts_old = mu_old.shape[1]
    for i, image in enumerate(img):
        for k in range(n_parts):
            cv2.drawMarker(image, (int(mu_scale[i][k][1]), int(mu_scale[i][k][0])), color_list[k],
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    img_old = np.ascontiguousarray(original)
    mu_scale_old = (mu_old + 1.) / 2. * img_old.shape[1]
    n_parts_old = mu_old.shape[1]
    for i, image in enumerate(img_old):
        for k in range(n_parts_old):
            cv2.drawMarker(image, (int(mu_scale_old[i][k][1]), int(mu_scale_old[i][k][0])), color_list[k],
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    with PdfPages(directory + '_predictions.pdf') as pdf:
        fig_head, axs_head = plt.subplots(bn // 4, 4, figsize=(30, 30))
        fig_head.suptitle("Overview", fontsize="x-large")
        for i in range(bn):
            axs_head[i // 4, i % 4].imshow(img[i])
            axs_head[i // 4, i % 4].axis('off')

        w, h = fig_head.canvas.get_width_height()
        # img = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

        # Plot Visualization
        part_map_overlay = torch.sum(part_map_old, dim=1).cpu().detach().numpy()
        heat_map_overlay = torch.sum(heat_map_old, dim=1).cpu().detach().numpy()
        fig_head2, axs_head = plt.subplots(4, 5, figsize=(20, 20))
        fig_head2.suptitle("Overview", fontsize="x-large")
        for i in range(4):
            axs_head[i, 0].imshow(org[index + i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 0].axis('off')
            axs_head[i, 1].imshow(1 - part_map_overlay[index + i], cmap='gray')
            axs_head[i, 1].axis('off')
            axs_head[i, 2].imshow(1 - heat_map_overlay[index + i], cmap='gray')
            axs_head[i, 2].axis('off')
            axs_head[i, 3].imshow(reconstruction[index + i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 3].axis('off')
            axs_head[i, 4].imshow(img_old[index + i])
            axs_head[i, 4].axis('off')

        # Plot Maps
        fig_shape, axs_shape = plt.subplots(6, 8, figsize=(20, 20))
        fig_shape.tight_layout()
        for i in range(6):
            for j in range(8):
                axs_shape[i, j].xaxis.set_major_locator(plt.NullLocator())
                axs_shape[i, j].yaxis.set_major_locator(plt.NullLocator())
        fig_shape.suptitle("Part Visualization", fontsize="x-large")
        color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine',
                      'green',
                      'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen', 'red']
        for i in range(n_parts_old):
            cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                            ['white', color_list[i]],
                                                            256)
            if 2 * (i % 3) == 2:
                axs_shape[int(i / 3), 2 * (i % 3) + 1].imshow(part_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
                axs_shape[int(i / 3), 2 * (i % 3) + 1 + 1].imshow(heat_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
            elif 2 * (i % 3) == 4:
                axs_shape[int(i / 3), 2 * (i % 3) + 2].imshow(part_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
                axs_shape[int(i / 3), 2 * (i % 3) + 1 + 2].imshow(heat_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
            else:
                axs_shape[int(i / 3), 2 * (i % 3)].imshow(part_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
                axs_shape[int(i / 3), 2 * (i % 3) + 1].imshow(heat_map_old[index][i].cpu().detach().numpy(), cmap=cmap)

        axs_shape[5, 6].imshow(part_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
        axs_shape[5, 7].imshow(heat_map_old[index][i].cpu().detach().numpy(), cmap=cmap)
        for i in range(6):
            axs_shape[i, 2].set_visible(False)
            axs_shape[i, 5].set_visible(False)
        axs_shape[5, 3].set_visible(False)
        axs_shape[5, 4].set_visible(False)
        plt.subplots_adjust(wspace=0, hspace=0)
        pdf.savefig(fig_head)
        pdf.savefig(fig_head2)
        pdf.savefig(fig_shape)

        fig_head.canvas.draw()
        fig_head2.canvas.draw()
        fig_shape.canvas.draw()
        plt.close('all')

    return img

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__ == '__main__':
    pass