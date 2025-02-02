import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class YouTubeVOSTestDataset(Dataset):
    """ YouTubeVOS dataset, structure:
    ├── all_frames
    │   └── valid_all_frames
    ├── train
    │   ├── Annotations
    │   └── JPEGImages
    ├── train_480p
    │   ├── Annotations
    │   └── JPEGImages
    └── valid
        ├── Annotations
        └── JPEGImages

    """
    def __init__(self, data_root, split, res=480):
        self.image_dir = path.join(data_root, 'all_frames', split+'_all_frames', 'JPEGImages')
        self.mask_dir = path.join(data_root, split, 'Annotations')

        self.videos = []
        self.shape = {}
        self.frames = {}

        vid_list = sorted(os.listdir(self.image_dir))
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(path.join(self.mask_dir, vid))[0]
            _mask = np.array(Image.open(path.join(self.mask_dir, vid, first_mask)).convert("P"))
            self.shape[vid] = np.shape(_mask)

        if res != -1:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(res, interpolation=InterpolationMode.BICUBIC),
            ])

            self.mask_transform = transforms.Compose([
                transforms.Resize(res, interpolation=InterpolationMode.NEAREST),
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
            ])

            self.mask_transform = transforms.Compose([
            ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video
        info['frames'] = self.frames[video]
        info['size'] = self.shape[video] # Real sizes
        info['gt_obj'] = {} # Frames with labelled objects

        vid_im_path = path.join(self.image_dir, video)
        vid_gt_path = path.join(self.mask_dir, video)

        frames = self.frames[video]

        images = []
        masks = []
        for i, f in enumerate(frames):
            img = Image.open(path.join(vid_im_path, f)).convert('RGB')
            images.append(self.im_transform(img))

            mask_file = path.join(vid_gt_path, f.replace('.jpg','.png'))
            if path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels
            else:
                # Mask not exists -> fill 0s
                # In val and test, only first frame is provided so from T=1 to end are just zeros masks.
                masks.append(np.zeros(self.shape[video]))

        images = torch.stack(images, 0) # (n_frames, 3, H, W)
        masks = np.stack(masks, 0) # (n_frames, H, W)

        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]
        info['label_convert'] = {}
        info['label_backward'] = {}
        idx = 1
        for l in labels:
            info['label_convert'][l] = idx
            info['label_backward'][idx] = l
            idx += 1
        masks = torch.from_numpy(all_to_onehot(masks, labels)).float() # (n_labels, n_frames, H, W)

        # Resize to 480p
        masks = self.mask_transform(masks) # (n_labels, n_frames, 480, 854) # or 853
        masks = masks.unsqueeze(2) # (n_labels, n_frames, 1, 480, 854)

        info['labels'] = labels

        data = {
            'rgb': images,  # (n_frames, 3, H, W) -> torch.Tensor
            'gt': masks,    # (n_labels, n_frames, 1, H, W) -> torch.Tensor
            'info': info,   # has labels, label mappings inside
        }

        return data

    def __len__(self):
        return len(self.videos)
