from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
import os
from glob import glob
import cv2
import pandas as pd
import numpy as np
import scipy.io
import json
import h5py


class DeepFashionDataset(Dataset):
    def __init__(self, size, train=True, mix=False):
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
        self.mix = mix

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

        # make additional keypoints
        if self.mix:
            extra_kp = 0.5 * self.size * torch.ones([1, 15, 2])
            keypoint = torch.cat([keypoint, extra_kp], dim=1)

        return image, keypoint


class Human36MDataset(Dataset):
    def __init__(self, size, mix=False):
        super(Human36MDataset, self).__init__()
        self.size = size
        self.img_shape = [1002, 1000, 3]
        self.base_path = "/export/scratch/compvis/datasets/human3.6M/"
        self.annot_path = "/export/home/phuber/BA/annot_small.json" if mix == False else "/export/home/phuber/BA/annot_very_small.json"
        with open(self.annot_path, "r") as json_file:
            f = json.load(json_file)
            self.images = [path for path in list(f['frame_path'])]
            self.keypoints = [np.flip(np.array(keypoint)).copy() for keypoint in list(f['keypoints'])]
            self.bboxes = [self.make_bbox(keypoint) for keypoint in self.keypoints]

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.mix = mix

    def make_bbox(self, keypoints):
        # bbox should have shape [left, up, right, low]
        h, w, c = self.img_shape
        padd = 30

        key_y = keypoints[:, 0]
        key_x = keypoints[:, 1]
        up = np.min(key_y).astype(int)
        up -= min(padd, up)
        low = np.max(key_y).astype(int)
        low += min(padd, h)
        left = np.min(key_x).astype(int)
        left -= min(padd, left)
        right = np.max(key_x).astype(int)
        right += min(padd, w)

        bbox = [left, up, right, low]
        return bbox

    def transform_keypoints(self, bbox, keypoint):
        h, w, c = self.img_shape
        h_, w_ = int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])
        long_side = max(h_, w_)
        # Don't use too small images
        if long_side < 50:
            return None, None, False

        # Don't just use bounding box but also a little bit more (else heads are cut off in TPS)
        padd_add = 30
        if long_side == h_:
            padd_long = min(h - h_, padd_add)
            if padd_long % 2 != 0:
                padd_long += 1
            padd_up = min(int(bbox[1]), padd_long / 2)
            padd_down = min(h - int(bbox[3]), padd_long - padd_up)
            if padd_up + padd_down < padd_long:
                padd_up += padd_long - (padd_up + padd_down)
            bbox[1] -= padd_up
            bbox[3] += padd_down
            h_, w_ = int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])
            long_side = max(h_, w_)
        else:
            padd_long = min(w - w_, padd_add)
            if padd_long % 2 != 0:
                padd_long += 1
            padd_left = min(int(bbox[0]), padd_long / 2)
            padd_right = min(h - int(bbox[2]), padd_long - padd_left)
            if padd_left + padd_right < padd_long:
                padd_left += padd_long - (padd_left + padd_right)
            bbox[0] -= padd_left
            bbox[2] += padd_right
            h_, w_ = int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])
            long_side = max(h_, w_)

        # Padd the sides to make it quadratic
        diff = max(h_, w_) - min(h_, w_)
        if diff % 2 != 0:
            diff += 1
        padding = diff / 2
        if long_side == h_:
            left_padding = min(padding, int(bbox[0]))
            right_padding = min(long_side - left_padding - w_, w - int(bbox[2]))
            if left_padding + w_ + right_padding < long_side:
                left_padding += long_side - (left_padding + w_ + right_padding)
            upper_padding = 0
            under_padding = 0
        else:
            upper_padding = min(padding, int(bbox[1]))
            under_padding = min(long_side - upper_padding - h_, h - int(bbox[3]))
            if upper_padding + h_ + under_padding < long_side:
                upper_padding += long_side - (upper_padding + h_ + under_padding)
            left_padding = 0
            right_padding = 0

        # Adjust boundaries
        left_bound = int(bbox[0]) - int(left_padding)
        right_bound = int(bbox[2]) + int(right_padding)
        upper_bound = int(bbox[1]) - int(upper_padding)
        under_bound = int(bbox[3]) + int(under_padding)
        boundaries = [left_bound, right_bound, upper_bound, under_bound]

        # Adjust keypoints
        keypoint[:, 1] -= left_bound
        keypoint[:, 0] -= upper_bound
        scale = self.size / long_side
        keypoint *= scale

        # Sometimes boundaries are outside the image -> skip these images
        if ((right_bound - left_bound) - (under_bound - upper_bound) != 0) or any(bound < 0 for bound in boundaries) \
                or right_bound > w or under_bound > h:
            return None, None, False

        return keypoint, boundaries, True


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Select Image
        image = cv2.imread(self.base_path + self.images[index])

        # Transform Image (crop and resize) and Keypoints
        init_bbox = self.bboxes[index]
        init_keypoints = self.keypoints[index]
        keypoint, bbox, _ = self.transform_keypoints(init_bbox, init_keypoints)
        left_bound, right_bound, upper_bound, under_bound = bbox
        image = image[upper_bound:under_bound, left_bound:right_bound, :]
        image = cv2.resize(image, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transforms(image)
        keypoint = self.transforms(keypoint)

        return image, keypoint


class PennAction(Dataset):
    def __init__(self, size, pose_req=None, action_req=None, mix=False):
        super(PennAction, self).__init__()
        self.size = size
        self.pose_req = pose_req
        self.action_req = action_req
        self.sequencepath = "/export/scratch/compvis/datasets/Penn_Action/frames/"
        self.labelpath = "/export/scratch/compvis/datasets/Penn_Action/labels/"
        sequences = natsorted(glob(os.path.join(self.sequencepath, "*")))
        annotations = natsorted(glob(os.path.join(self.labelpath, "*.mat")))
        poses = [[pose for pose in scipy.io.loadmat(annotations[i])['pose'] if annotations[i]] for i in
                 range(len(annotations))]
        poses = [pose for seq in poses for pose in seq]
        actions = [[action for action in scipy.io.loadmat(annotations[i])['action'] if annotations[i]] for i in
                 range(len(annotations))]
        actions = [action for seq in actions for action in seq]

        # Choose relevant poses
        if self.pose_req is not None:
            self.indices = [i for i in range(len(poses)) if poses[i] in self.pose_req]
            sequences = [sequences[i] for i in self.indices]
            annotations = [annotations[i] for i in self.indices]

        # Choose relevant actions
        if self.action_req is not None:
            self.indices = [i for i in range(len(actions)) if actions[i] in self.action_req]
            sequences = [sequences[i] for i in self.indices]
            annotations = [annotations[i] for i in self.indices]

        # Get Images and Keypoints
        images = [[img for img in natsorted(glob(os.path.join(sequence_path, "*.jpg")))] for sequence_path in sequences]
        images = [img for seq in images for img in seq]
        dimensions = [[dim for dim in scipy.io.loadmat(annotations[i])['shape'] if annotations[i]] for i in
                      range(len(annotations))]
        dimensions = [dim for seq in dimensions for dim in seq]
        bboxes = [[bb for bb in scipy.io.loadmat(annotations[i])['bbox'] if annotations[i]] for i in
                  range(len(annotations))]
        bboxes = [bb for seq in bboxes for bb in seq]
        kp_y = [[kp for kp in scipy.io.loadmat(annotations[i])['y'] if annotations[i]] for i in range(len(annotations))]
        kp_y = [kp for seq in kp_y for kp in seq]
        kp_x = [[kp for kp in scipy.io.loadmat(annotations[i])['x'] if annotations[i]] for i in range(len(annotations))]
        kp_x = [kp for seq in kp_x for kp in seq]
        keypoints = np.array([[[y, x] for (y, x) in zip(kp_y[i], kp_x[i])] for i in range(len(kp_x))])

        images_valid = []
        bboxes_valid = []
        boundaries_valid = []
        keypoints_t = []

        for index in range(len(images)):
            image, dimension, bbox, keypoint = images[index], dimensions[index], bboxes[index], keypoints[index]
            keypoint_t, boundaries, valid = self.transform_keypoints(dimension, bbox, keypoint)
            if not valid:
                continue
            kp_values = [j for i in keypoint_t for j in i]
            if min(kp_values) < 0 or max(kp_values) > self.size:
                continue
            else:
                images_valid.append(image)
                bboxes_valid.append(bbox)
                boundaries_valid.append(boundaries)
                keypoints_t.append(keypoint_t)

        self.images = np.array([images_valid])[0]
        self.bboxes = np.array([bboxes_valid])[0]
        self.keypoints = np.array([keypoints_t])[0]
        self.boundaries = boundaries_valid

        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.mix = mix

    def __len__(self):
        return len(self.images)

    def transform_keypoints(self, dimension, bbox, keypoint):
        h, w, c = dimension
        h_, w_ = int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])
        long_side = max(h_, w_)
        # Don't use too small images
        if long_side < 50:
            return None, None, False

        # Don't just use bounding box but also a little bit more (else heads are cut off in TPS)
        padd_add = 30
        if long_side == h_:
            padd_long = min(h-h_, padd_add)
            if padd_long % 2 != 0:
                padd_long += 1
            padd_up = min(int(bbox[1]), padd_long / 2)
            padd_down = min(h-int(bbox[3]), padd_long-padd_up)
            if padd_up + padd_down < padd_long:
                padd_up += padd_long - (padd_up + padd_down)
            bbox[1] -= padd_up
            bbox[3] += padd_down
            h_, w_ = int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])
            long_side = max(h_, w_)
        else:
            padd_long = min(w-w_, padd_add)
            if padd_long % 2 != 0:
                padd_long += 1
            padd_left = min(int(bbox[0]), padd_long / 2)
            padd_right = min(h-int(bbox[2]), padd_long-padd_left)
            if padd_left + padd_right < padd_long:
                padd_left += padd_long - (padd_left + padd_right)
            bbox[0] -= padd_left
            bbox[2] += padd_right
            h_, w_ = int(bbox[3]) - int(bbox[1]), int(bbox[2]) - int(bbox[0])
            long_side = max(h_, w_)

        # Padd the sides to make it quadratic
        diff = max(h_, w_) - min(h_, w_)
        if diff % 2 != 0:
            diff += 1
        padding = diff / 2
        if long_side == h_:
            left_padding = min(padding, int(bbox[0]))
            right_padding = min(long_side - left_padding - w_, w - int(bbox[2]))
            if left_padding + w_ + right_padding < long_side:
                left_padding += long_side - (left_padding + w_ + right_padding)
            upper_padding = 0
            under_padding = 0
        else:
            upper_padding = min(padding, int(bbox[1]))
            under_padding = min(long_side - upper_padding - h_, h - int(bbox[3]))
            if upper_padding + h_ + under_padding < long_side:
                upper_padding += long_side - (upper_padding + h_ + under_padding)
            left_padding = 0
            right_padding = 0

        # Adjust boundaries
        left_bound = int(bbox[0]) - int(left_padding)
        right_bound = int(bbox[2]) + int(right_padding)
        upper_bound = int(bbox[1]) - int(upper_padding)
        under_bound = int(bbox[3]) + int(under_padding)
        boundaries = [left_bound, right_bound, upper_bound, under_bound]

        # Adjust keypoints
        keypoint[:, 1] -= left_bound
        keypoint[:, 0] -= upper_bound
        scale = self.size / long_side
        keypoint *= scale

        # Sometimes boundaries are outside the image -> skip these images
        if ((right_bound - left_bound) - (under_bound - upper_bound) != 0) or any(bound < 0 for bound in boundaries) \
            or right_bound > w or under_bound > h:
            return None, None, False

        return keypoint, boundaries, True

    def __getitem__(self, index):
        # Select Image
        image = cv2.imread(self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Blur
        # filterSize = 5
        # image = cv2.medianBlur(image, filterSize)

        left_bound, right_bound, upper_bound, under_bound = self.boundaries[index]
        image = image[upper_bound:under_bound, left_bound:right_bound, :]
        image = cv2.resize(image, (self.size, self.size))
        keypoint = self.keypoints[index]

        image = self.transforms(image)
        keypoint = self.transforms(keypoint)

        if self.mix:
            extra_kp = 0.5 * self.size * torch.ones([1, 19, 2])
            keypoint = torch.cat([keypoint, extra_kp], dim=1)

        return image, keypoint


__datasets__ = {'deepfashion': DeepFashionDataset,
                'human36': Human36MDataset,
                'pennaction': PennAction}


def get_dataset(dataset_name):
    return __datasets__[dataset_name]


