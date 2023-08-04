from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset
from dataset.visor_dataset import VISORDataset
from dataset.egohos_dataset import EgoHOSDataset
from util.load_subset import load_sub_yv

# Test dataset
from dataset.yv_test_dataset import YouTubeVOSTestDataset
from dataset.davis_test_dataset import DAVISTestDataset
from dataset.visor_test_dataset import VISORTestDataset


def viz(data):
    import torch
    import numpy as np
    from PIL import Image
    """
    Visualise the data
    """
    def imwrite_mask(mask, out_path):
        davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
        davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                                [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                                [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                                [0, 64, 128], [128, 64, 128]]
        assert len(mask.shape) < 4 or mask.shape[0] == 1
        mask = Image.fromarray(mask, 'P')
        mask.putpalette(davis_palette.ravel())
        mask.save(out_path)

    frames = data['rgb'].numpy() # (3, 3, 384, 384)
    masks = data['cls_gt'] # (3, 1, 384, 384)
    boundaries = None
    if 'boundary' in data:
        boundaries = data['boundary']

    if type(masks) == torch.Tensor:
        masks = masks.numpy()
    for idx in range(3):
        img = np.round(frames[idx] * 255).astype(np.uint8)
        img = img.transpose(1, 2, 0) # (384, 384, 3)
        Image.fromarray(img).save('{}.jpg'.format(idx))

        mask = masks[idx].squeeze().astype(np.uint8) # (384, 384)
        imwrite_mask(mask, '{}.png'.format(idx))

        if boundaries:
            left = boundaries[idx][0].squeeze().astype(np.uint8)
            imwrite_mask(left, '{}_left.png'.format(idx))
            right = boundaries[idx][1].squeeze().astype(np.uint8)
            imwrite_mask(right, '{}_right.png'.format(idx))


# fss_dir = {'root': '/data/add_disk1/huyanh/Thesis/static/fss',
#            'method': 0}

# fss = StaticTransformDataset(**fss_dir)
# data = fss[0]
# print('DEBUG')

# davis_dir = {'im_root': '/data/add_disk1/huyanh/DAVIS/2017/trainval/JPEGImages/480p',
#              'gt_root': '/data/add_disk1/huyanh/DAVIS/2017/trainval/Annotations/480p',
#              'is_bl': False,
#              'max_jump': 5,
#              'subset': load_sub_yv('/data/add_disk1/huyanh/Thesis/STCN_VISOR/util/davis_subset.txt')}

# davis = VOSDataset(**davis_dir)
# data = davis[0]
# print('DEBUG')

# ytvos_dir = {'im_root': '/data/add_disk1/huyanh/Thesis/YouTube/train_480p/JPEGImages',
#              'gt_root': '/data/add_disk1/huyanh/Thesis/YouTube/train_480p/Annotations',
#              'is_bl': False,
#              'max_jump': 5,
#              'subset': load_sub_yv('/data/add_disk1/huyanh/Thesis/STCN_VISOR/util/yv_subset.txt')}

# ytvos = VOSDataset(**ytvos_dir)
# data = ytvos[0]
# print('DEBUG')

# visor_dir = {'im_root': '/data/add_disk1/huyanh/Thesis/VISOR_NO_AUG/VISOR_2022_YTVOS/train/JPEGImages',
#              'gt_root': '/data/add_disk1/huyanh/Thesis/VISOR_NO_AUG/VISOR_2022_YTVOS/train/Annotations',
#              'bound_root': '/data/add_disk1/huyanh/Thesis/VISOR_NO_AUG/VISOR_2022_YTVOS/train/Boundaries',
#              'skip_frame': False,
#              'include_hand': False}

# visor = VISORDataset(**visor_dir)
# data = visor[0]
# print('DEBUG')

# egohos_dir = {'root': '/data/add_disk1/huyanh/Thesis/EgoHOS_STATIC',
#               'subset': 'val'}
# egohos = EgoHOSDataset(**egohos_dir)
# data = egohos[0]
# print('DEBUG')


# ###### TEST DATASET ######
# davis_test_dir = {'root': '../DAVIS/2017/trainval',
#                   'imset': '2017/val.txt',
#                   'resolution': 480,
#                   'single_object': False,
#                   'target_name': None}
# davis_test = DAVISTestDataset(**davis_test_dir)
# data = davis_test[0]
# print('DEBUG')

# ytvos_test_dir = {'data_root': '../YouTube',
#                   'split': 'valid',
#                   'res': 480}

# ytvos_test = YouTubeVOSTestDataset(**ytvos_test_dir)
# data = ytvos_test[0]

visor_test_dir = {'root': '/data/add_disk1/huyanh/Thesis/VISOR_NO_AUG/VISOR_2022_YTVOS/val'}
visor_test = VISORTestDataset(**visor_test_dir)
data = visor_test[0]
print('DEBUG')















