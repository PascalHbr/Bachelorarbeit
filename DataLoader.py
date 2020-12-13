from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):
    def __init__(self, images, keypoints):
        super(ImageDataset, self).__init__()
        self.images = images
        self.keypoints = keypoints
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = self.images[index]
        keypoint = self.keypoints[index]

        image = self.transforms(image)
        keypoint = self.transforms(keypoint)

        return image, keypoint