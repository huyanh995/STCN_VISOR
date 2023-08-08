import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl,
                 subset=None, # manually left out some videos
                 ):
        # For YTVOS, choose 3464 over 3471 videos
        # For DAVIS, choose 60 over 90 videos

        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl # For BL30K set.

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < 3:
                # Skip videos with less than 3 frames
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # Data augmentation
        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=15, shear=10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BICUBIC)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
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

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames)-this_max_jump+1)
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
            f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                # Transforms apply the same for all pairs in the sequence
                # Because it contains random flip, which greatly affects model if applied to each frame independently.
                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)

                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                # Transforms apply randomly for each pair (RandomAffine)
                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im) # (3, 384, 384)
                this_gt = np.array(this_gt) # (384, 384)

                images.append(this_im) # list of tensors
                masks.append(this_gt) # list of np arrays

            images = torch.stack(images, 0)

            labels = np.unique(masks[0]) # (3, 3, H, W)
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)

            if len(labels) == 0:
                # No label in any frame -> Retry
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0) # (3, H, W)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:] # (3, 1, H, W)
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:] # (3, 1, H, W)
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=int) # same shape as masks
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,          # (N=3, 3, 384, 384)    -> torch.Tensor
            'gt': tar_masks,        # (N=3, 1, 384, 384)    -> np.array
            'cls_gt': cls_gt,       # (N=3, 384, 384)       -> np.array
            'sec_gt': sec_masks,    # (N=3, 1, 384, 384)    -> np.array
            'selector': selector,   # (2)                   -> torch.FloatTensor
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
