import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from architecture import softmax
from ops import get_heat_map, get_mu_and_prec
import random
import cv2
import h5py
import json
from json import JSONEncoder



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
    color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine', 'green',
                  'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen', 'red']

    nk = original_part_maps.shape[1]

    # Get Maps
    fmap_shape_norm = softmax(fmap_shape)
    mu_shape, L_inv_shape = get_mu_and_prec(fmap_shape_norm, device, L_inv_scale)
    heat_map_shape = get_heat_map(mu_shape, L_inv_shape, device, background=False)

    fmap_app_norm = softmax(fmap_app)
    mu_app, L_inv_app = get_mu_and_prec(fmap_app_norm, device, L_inv_scale)
    heat_map_app = get_heat_map(mu_app, L_inv_app, device, background=False)

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
        fig_shape, axs_shape = plt.subplots(9, 6, figsize=(8, 8))
        fig_shape.suptitle("Part Visualization Shape Stream", fontsize="x-large")
        for i in range(nk):
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

            if i == nk - 1:
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['white', 'black'],
                                                                256)
                axs_head[1, 1].imshow(overlay_shape.cpu().detach().numpy(), cmap=cmap)

        # Part Visualization Appearance Stream
        fig_app, axs_app = plt.subplots(9, 6, figsize=(8, 8))
        fig_app.suptitle("Part Visualization Appearance Stream", fontsize="x-large")
        for i in range(nk):
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

            if i == nk - 1:
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['white', 'black'],
                                                                256)
                axs_head[1, 3].imshow(overlay_app.cpu().detach().numpy(), cmap=cmap)

        pdf.savefig(fig_head)
        pdf.savefig(fig_shape)
        pdf.savefig(fig_app)

        fig_head.canvas.draw()
        w, h = fig_head.canvas.get_width_height()
        img = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

        plt.close('all')

    return img


def visualize_VAE(org, rotation, reconstruction, mu, heat_map, labels, show_labels=True):
    heat_map_overlay = torch.sum(heat_map, dim=1).cpu().detach().numpy()

    # Mark Keypoints
    original, mu = org.permute(0, 2, 3, 1).cpu().detach().numpy(), mu.cpu().detach().numpy()
    img = np.ascontiguousarray(original)
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

    fig_head, axs_head = plt.subplots(4, 4, figsize=(12, 12))
    fig_head.suptitle("Overview", fontsize="x-large")
    for i in range(4):
        axs_head[i, 0].imshow(org[i].permute(1, 2, 0).cpu().detach().numpy())
        axs_head[i, 1].imshow(heat_map_overlay[i], cmap='gray')
        axs_head[i, 2].imshow(img[i])
        axs_head[i, 3].imshow(reconstruction[i].permute(1, 2, 0).cpu().detach().numpy())

    fig_head.canvas.draw()
    w, h = fig_head.canvas.get_width_height()
    img = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

    plt.close('all')

    return img

def visualize_keypoints(img, fmap, labels, L_inv_scale, device, show_labels):
    bn, nk, h, w = fmap.shape

    # Make Heatmap Overlay
    fmap_norm = softmax(fmap)
    mu, L_inv = get_mu_and_prec(fmap_norm, device, L_inv_scale)
    heat_map = get_heat_map(mu, L_inv, device, )

    norm = torch.sum(heat_map, 1, keepdim=True) + 1
    heat_map = heat_map / norm

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
    bn, _, nk, _ = ground_truth.shape
    preds = ((prediction + 1.) / 2. * image_size).float().detach().cpu()
    gt = ground_truth[:, 0].float().detach().cpu()
    first_indices = torch.arange(bn)[:, None]

    # sort by y index
    indices_gt = gt[:, :, 0].sort()[1]
    gt_sorted = gt[first_indices, indices_gt]

    indices_preds = preds[:, :, 0].sort()[1]
    preds_sorted = preds[first_indices, indices_preds]

    distances = torch.zeros(1)
    for b in range(bn):
        preds_bn = preds_sorted[b]
        gt_bn = gt_sorted[b]
        for k in range(nk):
            best_distance = 1e7
            best_index = 1e7
            for i in range(len(preds_bn)):
                distance = torch.mean(torch.cdist(preds_bn[i].unsqueeze(0), gt_bn[k].unsqueeze(0), p=2.0))
                if distance < best_distance:
                    best_distance = distance
                    best_index = i
            # if len(preds_bn) > 1:
            #     preds_bn = torch.cat([preds_bn[:best_index], preds_bn[best_index + 1:]])
            distances += best_distance

    distance_norm = distances / (bn * nk * image_size)

    return distance_norm


def visualize_SAE(org, img_reconstr, mu, prec, part_map_norm, heat_map_norm, labels, directory, epoch, show_labels=True):
    bn = org.shape[0] // 2
    # Make Heatmaps
    heat_map_overlay = torch.sum(heat_map_norm, dim=1).cpu().detach().numpy()
    part_map_overlay = torch.sum(part_map_norm, dim=1).cpu().detach().numpy()

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
                           markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1, line_type=cv2.LINE_AA)

        if show_labels:
            for n in range(n_labels):
                cv2.drawMarker(image, (int(labels[i][n][1]), int(labels[i][n][0])), (0, 1., 0),
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1, line_type=cv2.LINE_AA)

    with PdfPages(directory + str(epoch) + '_summary.pdf') as pdf:
        fig_head, axs_head = plt.subplots(4, 9, figsize=(15, 15))
        fig_head.suptitle("Overview", fontsize="x-large")
        for i in range(4):
            axs_head[i, 0].imshow(org[:bn][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 0].axis('off')
            axs_head[i, 1].imshow(part_map_overlay[:bn][i], cmap='gray')
            axs_head[i, 1].axis('off')
            axs_head[i, 2].imshow(heat_map_overlay[:bn][i], cmap='gray')
            axs_head[i, 2].axis('off')
            axs_head[i, 3].imshow(img_reconstr[:bn][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 3].axis('off')
            axs_head[i, 4].imshow(img[:bn][i])
            axs_head[i, 4].axis('off')
            axs_head[i, 5].imshow(org[bn:][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 5].axis('off')
            axs_head[i, 6].imshow(part_map_overlay[bn:][i], cmap='gray')
            axs_head[i, 6].axis('off')
            axs_head[i, 7].imshow(heat_map_overlay[bn:][i], cmap='gray')
            axs_head[i, 7].axis('off')
            axs_head[i, 8].imshow(img_reconstr[bn:][i].permute(1, 2, 0).cpu().detach().numpy())
            axs_head[i, 8].axis('off')

        pdf.savefig(fig_head)

        fig_head.canvas.draw()
        w, h = fig_head.canvas.get_width_height()
        img = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

        plt.close('all')

    return img


def visualize_predictions(org, mu, directory):
    color_list = [(0,0,0), (255,255,255), (255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (174, 101, 0),
                  (192,192,192), (128,128,128), (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128), (0,0,128)]

    bn = org.shape[0]
    assert bn % 4 == 0
    original, mu = org.permute(0, 2, 3, 1).cpu().detach().numpy(), mu.cpu().detach().numpy()
    img = np.ascontiguousarray(original)
    mu_scale = (mu + 1.) / 2. * img.shape[1]
    n_parts = mu.shape[1]
    for i, image in enumerate(img):
        for k in range(n_parts):
            cv2.drawMarker(image, (int(mu_scale[i][k][1]), int(mu_scale[i][k][0])), color_list[k],
                           markerType=cv2.MARKER_TILTED_CROSS, markerSize=10, thickness=2, line_type=cv2.LINE_AA)

    with PdfPages(directory + 'predictions.pdf') as pdf:
        fig_head, axs_head = plt.subplots(bn // 4, 4, figsize=(15, 15))
        fig_head.suptitle("Overview", fontsize="x-large")
        for i in range(bn):
            axs_head[i // 4, i % 4].imshow(img[i])
            axs_head[i // 4, i % 4].axis('off')

        pdf.savefig(fig_head)

        fig_head.canvas.draw()
        w, h = fig_head.canvas.get_width_height()
        img = np.fromstring(fig_head.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape((w, h, 3))

        plt.close('all')

    return img

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

if __name__ == '__main__':
    pass
    base_path = "/export/scratch/compvis/datasets/human3.6M/"
    annot_path = base_path + "processed/all/annot.h5"

    with h5py.File(annot_path, "r") as f:
        random_indices = sorted(random.sample(range(1, 2108571), 12000))
        f_sub = {key: [x.decode('UTF-8') if isinstance(x, bytes) else x.tolist() for x in f[key][random_indices]] for key in f.keys()}
        with open('../very_annot_small.json', 'w') as fp:
            json.dump(f_sub, fp, cls=NumpyArrayEncoder)
        # f_sub = {key: f[key][random_indices] for key in f.keys()}
        # fname = '../annot_very_small.h5'
        # hdfdict.dump(f_sub, fname)


