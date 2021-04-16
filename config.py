import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # General configuration
    parser.add_argument('--name', default="trial1", type=str, help="name of the experiment", required=True)
    parser.add_argument('--gpu', nargs="+", type=int, required=True)
    parser.add_argument('--dataset', default='pennaction', help="name of the dataset")
    parser.add_argument('--module', type=int, choices=[1, 2, 3, 4], help="1: HG-UNet, 2:ViT-UNet"
                                                                         "3: HG-ViT, 4:ViT-ViT", required=True)
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--load_from_ckpt', default=False, type=bool)

    # General Hyperparameters
    parser.add_argument('--epochs', default=1000, type=int, help="number of epochs")
    parser.add_argument('--batch_size', default=12, type=int, help="batchsize")
    parser.add_argument('--lr', default=1e-4, type=float, help="learning rate of network")
    parser.add_argument('--reconstr_dim', default=256, type=int, help="dimension of reconstruction")
    parser.add_argument('--depth_s', default=4, type=int, help="depth of shape hourglass")
    parser.add_argument('--depth_a', default=1, type=int, help="depth of appearance hourglass")
    parser.add_argument('--residual_dim', default=256, type=int, help="neurons in residual module of the hourglass")
    parser.add_argument('--n_features', default=128, type=int, help="neurons of feature map layer")
    parser.add_argument('--p_dropout', default=0.2, type=float, help="dropout rate")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help="weight_decay")

    # Dataset Specific Hyperparameters
    parser.add_argument('--n_parts', default=17, type=int, help="number of keypoints")
    parser.add_argument('--background', default=False, type=bool, help="use additional background map")

    # Hyperparameters for Loss Function
    parser.add_argument('--L_mu', default=1.0, type=float, help="")
    parser.add_argument('--L_cov', default=0.1, type=float, help="")
    parser.add_argument('--L_rec', default=1.0, type=float, help="")

    # Fine Tuning Hyperparameters
    parser.add_argument('--l_2_scal', default=0.4, type=float, help="scale around part means that is considered for l2")
    parser.add_argument('--l_2_threshold', default=0., type=float, help="")
    parser.add_argument('--L_inv_scal', default=0.8, type=float, help="")
    parser.add_argument('--scal', default=0.95, type=float, help="default 0.6 sensible schedule [0.6, 0.6]")
    parser.add_argument('--fold_with_L_inv', default=True, type=bool, help="Detach L_inv for Folding, else detach mu")

    # Transformation Hyperparameters (shape)
    parser.add_argument('--tps_scal', default=0.08, type=float, help="sensible schedule [0.01, 0.08]")
    parser.add_argument('--rot_scal', default=0.3, type=float, help="sensible schedule [0.05, 0.6]")
    parser.add_argument('--off_scal', default=0.15, type=float, help="sensible schedule [0.05, 0.15]")
    parser.add_argument('--scal_var', default=0.05, type=float, help="sensible schedule [0.05, 0.2]")
    parser.add_argument('--augm_scal', default=1., type=float, help="sensible schedule [0.0, 1.]")

    # Transformation Hyperparameters (appearance)
    parser.add_argument('--contrast', default=0.15, type=float,  help="contrast variation")
    parser.add_argument('--brightness', default=0.15, type=float, help="brightness variation")
    parser.add_argument('--saturation', default=0.1, type=float, help="saturation variation")
    parser.add_argument('--hue', default=0.15, type=float,  help="hue variation")

    # ViT Hyperparameters (HG)
    parser.add_argument('--hg_patch_size', default=4, type=int, help="size of image patches")
    parser.add_argument('--hg_dim', default=256, type=int, help="dimension of patch embedding")
    parser.add_argument('--hg_depth', default=8, type=int, help="number of transformer blocks - for shape its 2*depth")
    parser.add_argument('--hg_heads', default=8, type=int, help="number of attention heads")
    parser.add_argument('--hg_mlp_dim', default=1024, type=int, help="dimension of the mlp layer")

    # ViT Hyperparameters (Decoder)
    parser.add_argument('--dec_patch_size', default=4, type=int, help="size of image patches")
    parser.add_argument('--dec_dim', default=256, type=int, help="dimension of patch embedding")
    parser.add_argument('--dec_depth', default=8, type=int, help="number of transformer blocks")
    parser.add_argument('--dec_heads', default=8, type=int, help="number of attention heads")
    parser.add_argument('--dec_mlp_dim', default=1024, type=int, help="dimension of the mlp layer")

    arg = parser.parse_args()
    return arg


def write_hyperparameters(r, save_dir):
    filename = save_dir + "/config.txt"
    with open(filename, "w") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line)
            print(line, file=input_file)
