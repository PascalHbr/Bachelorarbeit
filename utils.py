import torch
import matplotlib.pyplot as plt
import os
import csv
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from architecture_ops import softmax
from ops import get_heat_map, get_mu_and_prec


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir):
    model.load_state_dict(torch.load(model_save_dir + '/parameters'))
    return model


def load_images_from_folder():
    folder = "/export/scratch/compvis/datasets/deepfashion_vunet/train"
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


def load_deep_fashion_dataset():
    data_folder = "/export/scratch/compvis/datasets/deepfashion_inshop/Img/img/"
    csv_folder = "/export/scratch/compvis/datasets/compvis-datasets/deepfashion_allJointsVisible/"
    train_images = []
    test_images = []

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

    return train_images, test_images


def make_visualization(original, reconstruction, shape_transform, app_transform, fmap_shape,
                       fmap_app, directory, epoch, device, index=0):

    # Color List for Parts
    color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine',
                  'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen']
    # Get Maps
    fmap_shape_norm = softmax(fmap_shape)
    mu_shape, L_inv_shape = get_mu_and_prec(fmap_shape_norm, device, scal=5.)
    heat_map_shape = get_heat_map(mu_shape, L_inv_shape, device)

    fmap_app_norm = softmax(fmap_app)
    mu_app, L_inv_app = get_mu_and_prec(fmap_app_norm, device, scal=5.)
    heat_map_app = get_heat_map(mu_app, L_inv_app, device)

    with PdfPages(directory + '/summary/' + str(epoch) + '_summary.pdf') as pdf:
        # Make Head with Overview
        fig_head, axs_head = plt.subplots(3, 4, figsize=(12, 12))
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