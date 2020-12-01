import torch
import numpy as np
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages
from opt_einsum import contract
from torchvision.utils import save_image
from architecture_ops import softmax
from ops import get_heat_map, get_mu_and_prec


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.cpu().detach().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    return inp


def plot_tensor(tensor):
    np_tensor = convert_image_np(tensor)
    plt.imshow(np_tensor)
    plt.show()


def batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])     # does that work?
    colour = torch.tensor(colour, dtype=torch.float).to(device)
    colour_map = contract('bkij, kl -> blij', heat_map, colour)
    return colour_map


def save_heat_map(heat_map, directory):
    for i, part in enumerate(heat_map):
        part = part.unsqueeze(0)
        save_image(part, directory + '/parts/' + str(i) + '.png')


def np_batch_colour_map(heat_map, device):
    c = heat_map.shape[1]
    colour = []
    for i in range(c):
        colour.append(cm.hsv(float(i / c))[:3])
    np_colour = np.array(colour).to(device)
    colour_map = contract('bkij,kl->blij', heat_map, np_colour)
    return colour_map


def identify_parts(image, raw, n_parts, version):
    image_base = np.array(Image.fromarray(image[0]).resize((64, 64))) / 255.
    base = image_base[:, :, 0] + image_base[:, :, 1] + image_base[:, :, 2]
    directory = os.path.join('../images/' + str(version) + "/identify/")
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in range(n_parts):
        plt.imshow(raw[0, :, :, i] + 0.02 * base, cmap='gray')
        fname = directory + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')


def save(img, mu, counter, model_save_dir):
    batch_size, out_shape = img.shape[0], img.shape[1:3]
    marker_list = ["o", "v", "s", "|", "_"]
    directory = os.path.join(model_save_dir + '/predictions/landmarks/')
    if not os.path.exists(directory):
        os.makedirs(directory)
    s = out_shape[0] // 8
    n_parts = mu.shape[-2]
    mu_img = (mu + 1.) / 2. * np.array(out_shape)[0]
    steps = batch_size
    step_size = 1

    for i in range(0, steps, step_size):
        plt.imshow(img[i])
        for j in range(n_parts):
            plt.scatter(mu_img[i, j, 1], mu_img[i, j, 0],  s=s, marker=marker_list[np.mod(j, len(marker_list))], color=cm.hsv(float(j / n_parts)))

        plt.axis('off')
        fname = directory + str(counter) + '_' + str(i) + '.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()


def part_to_color_map(encoding_list, part_depths, size, device, square=True):
    part_maps = encoding_list[0][:, :part_depths[0], :, :]
    if square:
        part_maps = part_maps ** 4
    color_part_map = batch_colour_map(part_maps, device)
    color_part_map = torch.nn.Upsample(size=(size, size))(color_part_map)

    return color_part_map


def save_model(model, model_save_dir):
    torch.save(model.state_dict(), model_save_dir + '/parameters')


def load_model(model, model_save_dir):
    model.load_state_dict(torch.load(model_save_dir + '/parameters'))
    return model


def load_images_from_folder(stop=False):
    folder = "/export/scratch2/compvis_datasets/deepfashion_vunet/train/"
    images = []
    for i, filename in enumerate(os.listdir(folder)):
        img = plt.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
        if stop == True:
            if i == 3:
                break
    return images

def make_visualization(original, reconstruction, shape_transform, app_transform, fmap_shape, fmap_app, directory, epoch,
                       index=3):
    # Color List for Parts
    color_list = ['black', 'gray', 'brown', 'chocolate', 'orange', 'gold', 'olive', 'lawngreen', 'aquamarine',
                  'dodgerblue', 'midnightblue', 'mediumpurple', 'indigo', 'magenta', 'pink', 'springgreen']

    # Get Maps
    fmap_shape_norm = softmax(fmap_shape)
    mu_shape, L_inv_shape = get_mu_and_prec(fmap_shape_norm, 'cpu', scal=5.)
    heat_map_shape = get_heat_map(mu_shape, L_inv_shape, "cpu")

    fmap_app_norm = softmax(fmap_app)
    mu_app, L_inv_app = get_mu_and_prec(fmap_app_norm, 'cpu', scal=5.)
    heat_map_app = get_heat_map(mu_app, L_inv_app, "cpu")

    with PdfPages(directory + '/summary/' + str(epoch) + '_summary.pdf') as pdf:
        # Make Head with Overview
        fig_head, axs_head = plt.subplots(3, 4, figsize=(12, 12))
        fig_head.suptitle("Overview", fontsize="x-large")
        axs_head[0, 0].imshow(original[index].permute(1, 2, 0).numpy())
        axs_head[0, 1].imshow(app_transform[index].permute(1, 2, 0).numpy())
        axs_head[0, 2].imshow(shape_transform[index].permute(1, 2, 0).numpy())
        axs_head[0, 3].imshow(reconstruction[index].permute(1, 2, 0).numpy())

        axs_head[1, 0].imshow(app_transform[index].permute(1, 2, 0).numpy())
        axs_head[1, 2].imshow(shape_transform[index].permute(1, 2, 0).numpy())

        axs_head[2, 0].imshow(reconstruction[0].permute(1, 2, 0).numpy())
        axs_head[2, 1].imshow(reconstruction[1].permute(1, 2, 0).numpy())
        axs_head[2, 2].imshow(reconstruction[2].permute(1, 2, 0).numpy())
        axs_head[2, 3].imshow(reconstruction[3].permute(1, 2, 0).numpy())

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

            axs_shape[int(i / 2), (i % 2) * 3].imshow(fmap_shape[index][i].numpy(), cmap=cmap)
            axs_shape[int(i / 2), (i % 2) * 3 + 1].imshow(fmap_shape_norm[index][i].numpy(), cmap=cmap)
            axs_shape[int(i / 2), (i % 2) * 3 + 2].imshow(heat_map_shape[index][i].numpy(), cmap=cmap)

            if i == 15:
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['white', 'black'],
                                                                256)
                axs_head[1, 1].imshow(overlay_shape.numpy(), cmap=cmap)

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

            axs_app[int(i / 2), (i % 2) * 3].imshow(fmap_app[index][i].numpy(), cmap=cmap)
            axs_app[int(i / 2), (i % 2) * 3 + 1].imshow(fmap_app_norm[index][i].numpy(), cmap=cmap)
            axs_app[int(i / 2), (i % 2) * 3 + 2].imshow(heat_map_app[index][i].numpy(), cmap=cmap)

            if i == 15:
                cmap = colors.LinearSegmentedColormap.from_list('my_colormap',
                                                                ['white', 'black'],
                                                                256)
                axs_head[1, 3].imshow(overlay_app.numpy(), cmap=cmap)

        pdf.savefig(fig_head)
        pdf.savefig(fig_shape)
        pdf.savefig(fig_app)