import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="trial1", type=str,
                        help="name of the experiment", required=True)
    parser.add_argument('--gpu', default=0, type=int, required=True)
    # run setting
    parser.add_argument('--mode', default='train', choices=['train', 'predict'])
    parser.add_argument('--load_from_ckpt', default=False, type=bool)

    # dataset folder
    #parser.add_argument('--dataset', choices=dataset_map_train.keys(), required=True)

    # options
    parser.add_argument('--covariance', default=True, type=bool)
    parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
    parser.add_argument('--reconstr_dim', default=256, type=int, help="dimension of reconstruction")

    # modes
    parser.add_argument('--bn', default=12, type=int,  help="batchsize if not slim and 2 * batchsize if slim")
    parser.add_argument('--n_parts', default=16, type=int, help="number of parts")
    parser.add_argument('--n_features', default=64, type=int,  help="neurons of feature map layer")
    parser.add_argument('--n_c', default=3, type=int)
    parser.add_argument('--residual_dim', default=256, type=int,  help="neurons in residual module of the hourglass")
    parser.add_argument('--depth_s', default=4, type=int, help="depth of shape hourglass")
    parser.add_argument('--depth_a', default=1, type=int, help="depth of appearance hourglass")

    # loss multiplication constants
    parser.add_argument('--lr',  default=0.0001, type=float, help="learning rate of network")
    parser.add_argument('--weight_decay', default=5e-4, type=float, help="weight_decay")
    parser.add_argument('--L_mu', default=5., type=float, help="")
    parser.add_argument('--L_cov', default=0.1, type=float, help="")

    # tps parameters
    parser.add_argument('--l_2_scal', default=0.1, type=float, help="scale around part means that is considered for l2")
    parser.add_argument('--l_2_threshold', default=0.2, type=float, help="")
    parser.add_argument('--L_inv_scal', default=5.0, type=float, help="")
    parser.add_argument('--scal', default=1.0, type=float, nargs='+', help="default 0.6 sensible schedule [0.6, 0.6]")
    parser.add_argument('--tps_scal', default=0.05, type=float, nargs='+', help="sensible schedule [0.01, 0.08]")
    parser.add_argument('--rot_scal', default=0.2, type=float, nargs='+', help="sensible schedule [0.05, 0.6]")
    parser.add_argument('--off_scal', default=0.15, type=float, nargs='+', help="sensible schedule [0.05, 0.15]")
    parser.add_argument('--scal_var', default=0.05, type=float, nargs='+', help="sensible schedule [0.05, 0.2]")
    parser.add_argument('--augm_scal', default=1., type=float, nargs='+', help="sensible schedule [0.0, 1.]")

    #appearance parameters
    parser.add_argument('--contrast', default=0.5, type=float,  help="contrast variation")
    parser.add_argument('--brightness', default=0.3, type=float, help="brightness variation")
    parser.add_argument('--saturation', default=0.1, type=float, help="saturation variation")
    parser.add_argument('--hue', default=0.3, type=float,  help="hue variation")
    parser.add_argument('--p_flip', default=0., type=float, help="flip probability")
    parser.add_argument('--static', default=True)
    arg = parser.parse_args()
    return arg


def write_hyperparameters(r, save_dir):
    filename = save_dir + "/config.txt"
    with open(filename, "a") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line)
            print(line, file=input_file)