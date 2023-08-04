import os

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization
from dataset.util import all_to_onehot


class VISORTestDataset(Dataset):
    """
    For VISOR dataset, no boundaries are provided.
    Regular semi-supervised setting.
    """
    def __init__(self,
                 root: str) -> None:
        self.image_dir = os.path.join(root, 'JPEGImages')
        self.mask_dir = os.path.join(root, 'Annotations')

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.image_dir))
        # Pre-reading
        for vid in vid_list:
            frames = sorted(os.listdir(os.path.join(self.image_dir, vid)))
            self.frames[vid] = frames

            self.videos.append(vid)
            first_mask = os.listdir(os.path.join(self.mask_dir, vid))[0] # First mask is always provided
            _mask = np.array(Image.open(os.path.join(self.mask_dir, vid, first_mask)).convert('P'))

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
        info['gt_obj'] = {} # Frames with labelled objects

        vid_im_path = os.path.join(self.image_dir, video)
        vid_gt_path = os.path.join(self.mask_dir, video)

        frames = self.frames[video]

        images = []
        masks = []
        for i, frame in enumerate(frames):
            img = Image.open(os.path.join(vid_im_path, frame)).convert('RGB')
            images.append(self.im_transform(img))

            mask_file = os.path.join(vid_gt_path, frame.replace('.jpg','.png'))
            if os.path.exists(mask_file):
                masks.append(np.array(Image.open(mask_file).convert('P'), dtype=np.uint8))
                this_labels = np.unique(masks[-1])
                this_labels = this_labels[this_labels!=0]
                info['gt_obj'][i] = this_labels # save labels of provided masks

            else:
                # Mask not exists -> fill 0s
                # In val and test, only first frame is provided so from T=1 to end are just zeros masks.
                masks.append(np.zeros(self.shape[video]))

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        # Construct the forward and backward mapping table for labels
        # this is because YouTubeVOS's labels are sometimes not continuous
        # while we want continuous ones (for one-hot)
        # so we need to maintain a backward mapping table
        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels!=0]

        # Save hand label before mapping to continuous ones
        info['left_hand'] = 1 if 1 in labels else -1
        info['right_hand'] = 2 if 2 in labels else -1

        info['label_convert'] = {}
        info['label_backward'] = {}

        for idx, label in enumerate(labels, start=1):
            info['label_convert'][label] = idx
            info['label_backward'][idx] = label

        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()

        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        data = {
            'rgb': images,
            'gt': masks,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
