import torch
from torchvision import datasets, transforms
import numpy as np
from mnist_Model import ViT
import os
from utils import load_model, save_model
from mnist_config import write_hyperparameters, parse_args
import wandb
import torch.nn as nn
from dotmap import DotMap


def main(arg):
    # Set random seeds
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    np.random.seed(7)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(7)
    rng = np.random.RandomState(7)

    # Load arguments
    lr = arg.lr
    epochs = arg.epochs
    name = arg.name
    load_from_ckpt = arg.load_from_ckpt
    device = torch.device('cuda:' + str(arg.gpu) if torch.cuda.is_available() else 'cpu')

    # Make new directory
    model_save_dir = '../mnist/' + name
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # Save Hyperparameters
    write_hyperparameters(arg.toDict(), model_save_dir)

    # Load Model and Optimizer
    model = ViT(
                image_size=arg.image_size,
                patch_size=arg.patch_size,
                num_classes=arg.num_classes,
                dim=arg.dim,
                depth=arg.depth,
                heads=arg.heads,
                mlp_dim=arg.mlp_dim,
                dropout=0.1,
                emb_dropout=0.1,
                channels=1,
                ).to(device)
    if load_from_ckpt:
        model = load_model(model, model_save_dir, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()

    # Log with wandb
    wandb.init(project='MNIST', config=arg, name=arg.name)
    wandb.watch(model, log='all')

    # Load Dataset
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=arg.batch_size,
                                              shuffle=True, num_workers=2)
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(mnist_testset, batch_size=arg.batch_size,
                                             shuffle=False, num_workers=2)

    # Make Training
    with torch.autograd.set_detect_anomaly(False):
        for epoch in range(epochs + 1):
            # Train on Train Set
            model.train()
            for step, (x, target) in enumerate(trainloader):
                x, target = x.to(device), target.to(device)
                pred = model(x)
                loss = loss_fn(nn.LogSoftmax(dim=1)(pred), target)
                labels = torch.argmax(nn.Softmax(dim=1)(pred), dim=1)

                # Get Accuracy
                accuracy = (target == labels).sum().item() / target.size(0)

                # Zero out gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                wandb.log({"Training Loss": loss})
                wandb.log({"Training Accuracy": accuracy})

            # Evaluate on Test Set
            model.eval()
            with torch.no_grad():
                for step, (x, target) in enumerate(testloader):
                    x, target = x.to(device), target.to(device)
                    pred = model(x)
                    test_loss = loss_fn(nn.LogSoftmax(dim=1)(pred), target)
                    labels = torch.argmax(nn.Softmax(dim=1)(pred), dim=1)

                    # Get Accuracy
                    val_accuracy = (target == labels).sum().item() / target.size(0)

                    wandb.log({"Validation Loss": test_loss})
                    wandb.log({"Validation Accuracy": val_accuracy})
                    save_model(model, model_save_dir)


if __name__ == '__main__':
    arg = DotMap(vars(parse_args()))
    main(arg)



