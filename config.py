import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="trial1", type=str, help="name of the experiment", required=True)
    parser.add_argument('--gpu', type=int, default=None, required=True)
    parser.add_argument('--dataset', default='deepfashion', help="name of the dataset")

    # run setting
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--load_from_ckpt', default=False, type=bool)

    # options
    parser.add_argument('--covariance', default=True, type=bool)
    parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
    parser.add_argument('--reconstr_dim', default=256, type=int, help="dimension of reconstruction")

    # modes
    parser.add_argument('--batch_size', default=8, type=int,  help="batchsize if not slim and 2 * batchsize if slim")
    parser.add_argument('--n_parts', default=16, type=int, help="number of parts")
    parser.add_argument('--n_features', default=128, type=int,  help="neurons of feature map layer")
    parser.add_argument('--n_c', default=3, type=int)
    parser.add_argument('--residual_dim', default=256, type=int,  help="neurons in residual module of the hourglass")
    parser.add_argument('--depth_s', default=4, type=int, help="depth of shape hourglass")
    parser.add_argument('--depth_a', default=1, type=int, help="depth of appearance hourglass")

    # loss multiplication constants
    parser.add_argument('--lr',  default=1e-3, type=float, help="learning rate of network")
    parser.add_argument('--p_dropout', default=0.2, type=float, help="dropout rate")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help="weight_decay")
    parser.add_argument('--L_mu', default=5., type=float, help="")
    parser.add_argument('--L_cov', default=0.1, type=float, help="")
    parser.add_argument('--L_rec', default=1.0, type=float, help="")

    # tps parameters
    parser.add_argument('--fold_with_shape', default=True, type=bool, help="fold with shape or with mu")
    parser.add_argument('--l_2_scal', default=0.1, type=float, help="scale around part means that is considered for l2")
    parser.add_argument('--l_2_threshold', default=0.05, type=float, help="")
    parser.add_argument('--L_inv_scal', default=0.8, type=float, help="")
    parser.add_argument('--scal', default=0.9, type=float, help="default 0.6 sensible schedule [0.6, 0.6]")
    parser.add_argument('--tps_scal', default=0.05, type=float, help="sensible schedule [0.01, 0.08]")
    parser.add_argument('--rot_scal', default=0.1, type=float, help="sensible schedule [0.05, 0.6]")
    parser.add_argument('--off_scal', default=0.15, type=float, help="sensible schedule [0.05, 0.15]")
    parser.add_argument('--scal_var', default=0.05, type=float, help="sensible schedule [0.05, 0.2]")
    parser.add_argument('--augm_scal', default=1., type=float, help="sensible schedule [0.0, 1.]")

    # appearance parameters
    parser.add_argument('--contrast', default=0.5, type=float,  help="contrast variation")
    parser.add_argument('--brightness', default=0.3, type=float, help="brightness variation")
    parser.add_argument('--saturation', default=0.1, type=float, help="saturation variation")
    parser.add_argument('--hue', default=0.3, type=float,  help="hue variation")
    parser.add_argument('--static', default=True)

    # transformer parameters
    parser.add_argument('--t_patch_size', default=16, type=int, help="size of image patches")
    parser.add_argument('--t_dim', default=256, type=int, help="dimension of patch embedding")
    parser.add_argument('--t_depth', default=8, type=int, help="number of transformer blocks")
    parser.add_argument('--t_heads', default=8, type=int, help="number of attention heads")
    parser.add_argument('--t_mlp_dim', default=256, type=int, help="dimension of the mlp layer")
    parser.add_argument('--t_n_token', default=1, type=int, help="number of cls tokens")
    parser.add_argument('--t_use_first', default=False, help="bool if using cls tokens or not")

    # GSA parameters
    parser.add_argument('--gsa_dim', default=256, type=int, help="")
    parser.add_argument('--gsa_dim_out', default=256, type=int, help="")
    parser.add_argument('--gsa_dim_key', default=32, type=int, help="")
    parser.add_argument('--gsa_heads', default=8, type=int, help="")
    parser.add_argument('--gsa_length', default=256, type=int, help="")

    arg = parser.parse_args()
    return arg


def write_hyperparameters(r, save_dir):
    filename = save_dir + "/config.txt"
    with open(filename, "w") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line)
            print(line, file=input_file)
