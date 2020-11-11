from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import kornia.augmentation as K
from transformations import tps_parameters, make_input_tps_param, ThinPlateSpline


class ImageDataset(Dataset):
    def __init__(self, images, arg):
        super(ImageDataset, self).__init__()
        self.images = images
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                              ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]
        image = self.transforms(image)

        return image


class ImageDataset2(Dataset):
    def __init__(self, images, arg):
        super(ImageDataset2, self).__init__()
        self.images = images
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])
                                              ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]
        original = self.transforms(image)

        return original
