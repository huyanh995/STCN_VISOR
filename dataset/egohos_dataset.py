"""
Mask labels:
1 - left hand
2 - right hand
3 - direct contact object of left hand
4 - direct contact object of right hand
5 - direct contact object of both hands

6 - indirect contact object of left hand
7 - indirect contact object of right hand
8 - indirect contact object of both hands
"""
import time
import random
import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.tps import random_tps_warp
from dataset.reseed import reseed


class EgoHOSDataset(Dataset):
    """
    From EgoHOS static images and masks, generate pseudo VOS
    by apply random transforms.
    """

    def __init__(self,
                root: str,
                subset: str,
                prob: float = 0.2,
                ) -> None:
        assert subset in ['train', 'val']
        self.prob = prob
        self.im_root = os.path.join(root, subset, 'JPEGImages')
        self.gt_root = os.path.join(root, subset, 'Annotations')
        self.bound_root = os.path.join(root, subset, 'Boundaries')

        self.im_list = [im for im in os.listdir(self.im_root) if '.jpg' in im]

        print(f'{len(self.im_list)} images found in EgoHOS {subset} set')

        # Transformations
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0), # No hue change here as that's not realistic
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
            transforms.Resize(384, InterpolationMode.BICUBIC),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=20, scale=(0.9,1.1), shear=10, interpolation=InterpolationMode.BICUBIC, fill=0),
            transforms.Resize(384, InterpolationMode.NEAREST),
            transforms.RandomCrop((384, 384), pad_if_needed=True, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.05, 0.05, 0.05),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=im_mean),
            transforms.RandomHorizontalFlip(),
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0, scale=(0.8, 1.5), fill=0),
            transforms.RandomHorizontalFlip(),
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

        self.final_gt_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, idx: int) -> dict:
        file_name = self.im_list[idx]
        img = Image.open(os.path.join(self.im_root, file_name)).convert('RGB') # (480, 854, 3)
        mask = Image.open(os.path.join(self.gt_root, file_name).replace('.jpg', '.png')).convert('P') # (480, 854)
        boundary = Image.open(os.path.join(self.bound_root, file_name).replace('.jpg', '_boundary.png')).convert('RGB') # (480, 854, 3)

        sequence_seed = np.random.randint(2147483647)

        images, masks, boundaries = [], [], []

        for _ in range(3):
            reseed(sequence_seed)
            this_im = self.all_im_dual_transform(img)
            this_im = self.all_im_lone_transform(this_im)

            reseed(sequence_seed)
            this_gt = self.all_gt_dual_transform(mask)
            reseed(sequence_seed)
            this_cb = self.all_gt_dual_transform(boundary)

            pairwise_seed = np.random.randint(2147483647)
            reseed(pairwise_seed)
            this_im = self.pair_im_dual_transform(this_im)
            this_im = self.pair_im_lone_transform(this_im) # (384, 384, 3)
            reseed(pairwise_seed)
            this_gt = self.pair_gt_dual_transform(this_gt) # (384, 384)
            reseed(pairwise_seed)
            this_cb = self.pair_gt_dual_transform(this_cb) # (384, 384, 3)

            # Not using TPS for now
            this_im = self.final_im_transform(this_im) # (3, 384, 384)
            this_gt = np.array(this_gt) # (384, 384)
            this_cb = np.array(this_cb)[:, :, :2] / 128.0
            this_cb = this_cb.transpose(2, 0, 1) # (2, 384, 384)
            # this_gt = self.final_gt_transform(this_gt) # (1, 384, 384)
            # this_cb = self.final_gt_transform(this_cb) # (3, 384, 384)

            images.append(this_im)
            masks.append(this_gt)
            boundaries.append(this_cb)

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)
        boundary_masks = np.stack(boundaries, 0)

        labels = np.unique(masks[0]) # there is a chance that object disappears after those transformations
        labels = labels[labels != 0]

        selector = []

        left_hand_masks = (masks == 1).astype(np.float32)[:, np.newaxis, :, :]
        right_hand_masks = (masks == 2).astype(np.float32)[:, np.newaxis, :, :]

        hand_labels = labels[labels <= 2]
        target_object = np.random.choice(hand_labels) if len(hand_labels) > 0 else -1

        # Choose second object
        p = random.random()

        if target_object + 2 in labels:
            second_object = target_object + 2
            selector = [1, 1]
        elif 5 in labels:
            second_object = 5
            selector = [1, 1]
        else:
            second_object = -1
            selector = [1, 0]
        boundary_masks = boundary_masks[:, target_object - 1, :, :].astype(np.float32)

        # Swap hand and object based on prob
        if p < self.prob:
            if target_object == 1 and 4 in labels:
                second_object = 4
                selector = [1, 1]
                boundary_masks = np.zeros_like(boundary_masks)

            elif target_object == 2 and 3 in labels:
                second_object = 3
                selector = [1, 1]
                boundary_masks = np.zeros_like(boundary_masks)

        # object_labels = labels[labels > 2]
        # target_object = -1
        # second_object = -1
        # if object_labels:
        #     target_object = np.random.choice(object_labels)
        #     if len(object_labels) > 1:
        #         second_object = np.random.choice(object_labels[object_labels != target_object])
        #         selector += [1, 1]
        #     else:
        #         selector += [1, 0]
        # else:
        #     selector += [0, 0]

        selector = torch.FloatTensor(selector)

        target_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :]
        second_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :]

        cls_gt = np.zeros((3, 384, 384), dtype=int)
        cls_gt[target_masks[:, 0] > 0.5] = 1
        cls_gt[second_masks[:, 0] > 0.5] = 2

        info = {}
        info['name'] = self.im_list[idx]
        data = {
            'rgb': images,                  # (N=3, 3, 384, 384)    -> torch.tensor
            'gt': target_masks,             # (N=3, 1, 384, 384)    -> np.array
            'cls_gt': cls_gt,               # (N=3, 384, 384)       -> np.array
            'sec_gt': second_masks,         # (N=3, 1, 384, 384)    -> np.array
            'left_hand': left_hand_masks,   # (N=3, 1, 384, 384)    -> np.array
            'right_hand': right_hand_masks, # (N=3, 1, 384, 384)    -> np.array
            'boundary_gt': boundary_masks,     # (N=3, 2, 384, 384)    -> np.array
            'selector': selector,           # (4, )                 -> torch.tensor
            'info': info
        }

        return data

    def __len__(self) -> int:
        return len(self.im_list)





