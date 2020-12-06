from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, images):
        super(ImageDataset, self).__init__()
        self.images = images
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]
        image = self.transforms(image)

        return image