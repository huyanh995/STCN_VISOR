"""
DataSet class for VISOR dataset.
VISOR has:
- RGB frames
- Annotations mask (1 channel)
- Contact boundary mask (3 channels)

- There will be at most 4 objects in a frame:
    - 1 and 2 for left and right hand
    - Then left object and right object

"""

import os

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

class VISORDataset(Dataset):
    """
    VISOR should follow the same format as YouTubeVOS
    with additional contact boundary mask
    """
    def __init__(self,
                 im_root: str,
                 gt_root: str,
                 bound_root: str,
                 skip_frame: bool,
                 include_hand: bool,
                 ) -> None:
        self.im_root = im_root
        self.gt_root = gt_root
        self.bound_root = bound_root
        self.skip_frame = skip_frame
        self.include_hand = include_hand

        self.videos = []
        self.frames = {} # vid : [frames]

        vid_list = sorted(os.listdir(self.im_root))
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                # actually VISOR sequences all have more than 3 frames
                # just for safety
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        # Data augmentation
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BICUBIC)
        ])

        self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
        ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])


    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = os.path.join(self.im_root, video)
        vid_gt_path = os.path.join(self.gt_root, video)
        vid_cb_path = os.path.join(self.bound_root, video) # contact boundary

        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = []
            if not self.skip_frame:
                # Randomly choose 3 consecutive frames, duplicated at most 2 of them.
                start_idx = np.random.randint(0, len(frames) - 1)
                f1_idx = start_idx + np.random.randint(0, 2)
                f1_idx = min(f1_idx, len(frames) - 1)

                f2_idx = f1_idx + np.random.randint(0, 2) if f1_idx != start_idx else f1_idx + 1
                f2_idx = min(f2_idx, len(frames) - 1)

            else:
                this_max_jump = min(len(frames), len(frames) // 6) # most VISOR has 6 frames in total.
                start_idx = np.random.randint(0, len(frames) - this_max_jump + 1)

                f1_idx = start_idx + np.random.randint(0, this_max_jump + 1) + 1
                f1_idx = min(f1_idx, len(frames) - this_max_jump, len(frames) - 1)

                f2_idx = f1_idx + np.random.randint(this_max_jump + 1) + 1
                f2_idx = min(f2_idx, len(frames) - this_max_jump // 2, len(frames) - 1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                frames_idx = frames_idx[::-1] # reverse order

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            boundaries = []

            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                cb_name = frames[f_idx][:-4] + '_boundary.png'

                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(os.path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im) # RandomHorizontalFlip + RandomResizedCrop
                this_im = self.all_im_lone_transform(this_im) # ColorJitter + RandomGrayscale

                reseed(sequence_seed)
                this_gt = Image.open(os.path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt) # RandomHorizontalFlip + RandomResizedCrop

                reseed(sequence_seed)
                this_cb = Image.open(os.path.join(vid_cb_path, cb_name)).convert('RGB') # (values are 128s)
                this_cb = self.all_gt_dual_transform(this_cb) # RandomHorizontalFlip + RandomResizedCrop

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im) # RandomAffine
                this_im = self.pair_im_lone_transform(this_im) # ColorJitter
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt) # RandomAffine
                reseed(pairwise_seed)
                this_cb = self.pair_gt_dual_transform(this_cb) # RandomAffine

                # Final transforms into tensors
                this_im = self.final_im_transform(this_im) # (3, 384, 384)
                this_gt = np.array(this_gt) # (384, 384)
                this_cb = np.array(this_cb)[:, :, :2] / 128.0 # remove last blank channel
                this_cb = this_cb.transpose(2, 0, 1) # (2, H, W)

                ####
                images.append(this_im)
                masks.append(this_gt)
                boundaries.append(this_cb)

            images = torch.stack(images, dim=0) # (3, 3, 384, 384)
            labels = np.unique(masks[0]) # NOTE: may not be just the first mask
            labels = labels[labels != 0] # remove background

            """
            Logic note:
            if include_hand, which means model will learn to segment hand, then no object in
            hand is fine.
            Unlike YouTubeVOS, there are some frames that masks are completely empty.
            but it's still useful and need to learn.
            """
            try_again = ((self.include_hand and len(labels) == 0)
                         or (not self.include_hand and len(labels[labels > 2]) == 0))

            if try_again:
                # no hands nor objects, try again
                target_object = -1
                has_second_object = False
                has_left_hand = False
                has_right_hand = False
                trials += 1

            else:
                # There are case only hand masks and no object
                object_labels = labels[labels > 2]
                target_object = np.random.choice(object_labels)
                has_second_object = len(object_labels) > 1
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)

                has_left_hand = 1 in labels
                has_right_hand = 2 in labels

                break


        # Construct data to tensors
        masks = np.stack(masks, axis=0) # (3, H, W)
        target_masks = (masks == target_object).astype(np.float32)[:, np.newaxis, :, :] # (3, 1, H, W)
        boundary_masks = np.stack(boundaries, axis=0) # (3, 2, H, W)

        hand_selector = []
        object_selector = []

        left_hand_masks = np.zeros_like(target_masks)
        right_hand_masks = np.zeros_like(target_masks)

        if self.include_hand:
            if has_left_hand:
                left_hand_masks = (masks == 1).astype(np.float32)[:, np.newaxis, :, :]
                hand_selector.append(1)
            else:
                left_hand_masks = np.zeros_like(target_masks)
                hand_selector.append(0)

            if has_right_hand:
                right_hand_masks = (masks == 2).astype(np.float32)[:, np.newaxis, :, :]
                hand_selector.append(1)
            else:
                right_hand_masks = np.zeros_like(target_masks)
                hand_selector.append(0)

        if has_second_object:
            second_masks = (masks == second_object).astype(np.float32)[:, np.newaxis, :, :] # (3, 1, H, W)
            object_selector = [1, 1]
        else:
            second_masks = np.zeros_like(target_masks)
            object_selector = [1, 0]

        selector = torch.FloatTensor(hand_selector + object_selector) # (4, )

        cls_gt = np.zeros((3, 384, 384), dtype=int)
        if self.include_hand:
            cls_gt[left_hand_masks[:, 0] > 0.5] = 1
            cls_gt[right_hand_masks[:, 0] > 0.5] = 2
            cls_gt[target_masks[:, 0] > 0.5] = 3
            cls_gt[second_masks[:, 0] > 0.5] = 4
        else:
            cls_gt[target_masks[:, 0] > 0.5] = 1
            cls_gt[second_masks[:, 0] > 0.5] = 2


        data = {
            'rgb': images,                  # (N=3, 3, 384, 384)    -> torch.tensor
            'gt': target_masks,             # (N=3, 1, 384, 384)    -> np.array
            'cls_gt': cls_gt,               # (N=3, 384, 384)       -> np.array
            'sec_gt': second_masks,         # (N=3, 1, 384, 384)    -> np.array
            'left_hand': left_hand_masks,   # (N=3, 1, 384, 384)    -> np.array
            'right_hand': right_hand_masks, # (N=3, 1, 384, 384)    -> np.array
            'boundary': boundary_masks,     # (N=3, 2, 384, 384)    -> np.array
            'selector': selector,           # (4, )                 -> torch.tensor
            'info': info
        }

        return data


    def __len__(self) -> int:
        return len(self.videos)
