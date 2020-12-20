import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default="trial1", type=str, help="name of the experiment", required=True)
    parser.add_argument('--gpu', default=None, required=True)
    parser.add_argument('--load_from_ckpt', default=False, type=bool)

    # run setting

    parser.add_argument('--epochs', default=100, type=int, help="number of epochs")
    parser.add_argument('--batch_size', default=64, type=int,  help="batchsize if not slim and 2 * batchsize if slim")
    parser.add_argument('--lr',  default=1e-3, type=float, help="learning rate of network")

    # transformer parameters
    parser.add_argument('--image_size', default=28, type=int, help="number of cls tokens")
    parser.add_argument('--patch_size', default=2, type=int, help="size of image patches")
    parser.add_argument('--num_classes', default=10, type=int, help="dimension of patch embedding")
    parser.add_argument('--dim', default=1024, type=int, help="number of transformer blocks")
    parser.add_argument('--depth', default=6, type=int, help="number of attention heads")
    parser.add_argument('--heads', default=16, type=int, help="dimension of the mlp layer")
    parser.add_argument('--mlp_dim', default=512, type=int, help="number of cls tokens")

    arg = parser.parse_args()
    return arg

def write_hyperparameters(r, save_dir):
    filename = save_dir + "/config.txt"
    with open(filename, "w") as input_file:
        for k, v in r.items():
            line = '{}, {}'.format(k, v)
            print(line)
            print(line, file=input_file)
