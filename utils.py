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
import h5py
import scipy.io


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


def visualize_keypoints(img, fmap, labels, L_inv_scale, device, show_labels):
    bn, nk, h, w = fmap.shape

    # Make Heatmap Overlay
    fmap_norm = softmax(fmap)
    mu, L_inv = get_mu_and_prec(fmap_norm, device, L_inv_scale)
    heat_map = get_heat_map(mu, L_inv, device)

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
    bn, nk, _ = prediction.shape
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
            if len(preds_bn) > 1:
                preds_bn = torch.cat([preds_bn[:best_index], preds_bn[best_index + 1:]])
            distances += best_distance

    distance_norm = distances / (bn * nk * image_size)

    return distance_norm


def crop_and_resize(image, bbox, keypoint, size=256):
    image_org = image.copy()
    for i in range(13):
        cv2.drawMarker(image_org, (int(keypoint[i][1]), int(keypoint[i][0])), (1., 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1, line_type=cv2.LINE_AA)
    plt.imshow(image_org)
    plt.show()

    h, w, c = image.shape
    image_crop = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
    h_, w_, c_ = image_crop.shape
    long_side = max(h_, w_)
    diff = max(h_, w_) - min(h_, w_)
    if diff % 2 != 0:
        diff += 1
    padding = diff / 2
    if long_side == h_:
        left_padding = min(padding, int(bbox[0]))
        right_padding = min(long_side - left_padding - w_, w - int(bbox[2]))
        if left_padding + w_ + right_padding < long_side:
            left_padding += long_side - (left_padding + w_ + right_padding)
        upper_padding = 0
        under_padding = 0
    else:
        upper_padding = min(padding, int(bbox[1]))
        under_padding = min(long_side - upper_padding- h_, h - int(bbox[3]))
        if upper_padding + h_ + under_padding < long_side:
            upper_padding += long_side - (upper_padding + h_ + under_padding)
        left_padding = 0
        right_padding = 0

    left_bound = int(bbox[0]) - int(left_padding)
    right_bound = int(bbox[2]) + int(right_padding)
    upper_bound = int(bbox[1]) - int(upper_padding)
    under_bound = int(bbox[3]) + int(under_padding)

    keypoint[:, 1] -= left_bound
    keypoint[:, 0] -= upper_bound

    scale = size / long_side
    keypoint *= scale

    image_crop = image[upper_bound:under_bound, left_bound:right_bound, :]
    image_crop = cv2.resize(image_crop, (size, size))
    for i in range(13):
        cv2.drawMarker(image_crop, (int(keypoint[i][1]), int(keypoint[i][0])), (1., 0, 0),
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=1, line_type=cv2.LINE_AA)
    plt.imshow(image_crop)
    plt.show()

    return image_crop, keypoint



if __name__ == '__main__':
    sequencepath = "/export/scratch/compvis/datasets/Penn_Action/frames/"
    labelpath = "/export/scratch/compvis/datasets/Penn_Action/labels/"
    sequences = natsorted(glob(os.path.join(sequencepath, "*")))
    annotations = natsorted(glob(os.path.join(labelpath, "*.mat")))

    for i, img_path in enumerate(sequences):
        image = cv2.imread(img_path + "/000001.jpg")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        name = annotations[i]
        print(name)
        dic = scipy.io.loadmat(annotations[i])
        nframes = dic['nframes'][0][0]
        # dic['shape'] = [image.shape for i in range(nframes)]
        # scipy.io.savemat(name, dic, oned_as='row')


    # Select Image
    # index = 55000
    # image = cv2.imread(images[index])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # keypoint = keypoints[index].reshape(13, 2)
    # bbox = bboxes[index]
