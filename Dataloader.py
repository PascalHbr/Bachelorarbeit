from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
import os
from glob import glob
import cv2
import pandas as pd
import numpy as np



class DeepFashionDataset(Dataset):
    def __init__(self, size, train=True):
        super(DeepFashionDataset, self).__init__()
        self.size = size
        self.train = train
        self.basepath = "/export/scratch/compvis/datasets/deepfashion_inshop/Img/img/"
        self.csv_path = "/export/scratch/compvis/datasets/compvis-datasets/deepfashion_allJointsVisible/data_"
        self.annotations = "/export/scratch/compvis/datasets/compvis-datasets/deepfashion_allJointsVisible/data_"
        if self.train:
            subdir_name = "train"
        else:
            subdir_name = "test"
        self.img_path = pd.read_csv(os.path.join(self.csv_path + subdir_name + ".csv"))['filename'].tolist()
        self.keypoints = np.flip(np.array(pd.read_json(os.path.join(self.annotations + subdir_name + ".json"))['keypoints'].tolist()), 2).copy()
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        # Select Image
        image = cv2.imread(os.path.join(self.basepath, self.img_path[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = self.transforms(image)

        #Select Keypoint
        keypoint = self.keypoints[index]
        keypoint = self.transforms(keypoint)

        return image, keypoint


class Human3MDataset(Dataset):
    def __init__(self, size, train=True):
        super(Human3MDataset, self).__init__()
        self.size = size
        self.train = train
        self.basepath = "/export/scratch/compvis/datasets/human3M_lorenz19"
        if self.train:
            subdir_name = "train"
        else:
            subdir_name = "test"
        self.datafiles = natsorted(glob(os.path.join(self.basepath, subdir_name, "*", "*", "*.jpg")))
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, index):
        # Select Image
        image = cv2.imread(self.datafiles[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = self.transforms(image)

        return image, image
