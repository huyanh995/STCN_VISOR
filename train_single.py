import os
import datetime
from os import path
import math

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from model.model import STCNModelSingle
from dataset.static_dataset import StaticTransformDataset
from dataset.vos_dataset import VOSDataset

from util.logger import TensorboardLogger
from util.hyper_para import HyperParameters
from util.load_subset import load_sub_davis, load_sub_yv

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
"""
Initial setup
"""
# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

# Parse command line arguments
para = HyperParameters()
para.parse()

"""
Model related
"""
model = STCNModelSingle(para).train()
# Load pretrained model if needed
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print('Previously trained model loaded!')
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

"""
Dataloader related
"""
# To re-seed the randomness everytime we start a worker
local_rank = 0
def worker_init_fn(worker_id):
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank*100)

def construct_loader(dataset):
    train_loader = DataLoader(dataset, para['batch_size'], num_workers=para['num_workers'],
                            worker_init_fn=worker_init_fn, drop_last=True, pin_memory=True)
    return None, train_loader

def renew_vos_loader(max_skip):
    # //5 because we only have annotation for every five frames
    yv_dataset = VOSDataset(path.join(yv_root, 'JPEGImages'),
                        path.join(yv_root, 'Annotations'), max_skip//5, is_bl=False, subset=load_sub_yv())
    davis_dataset = VOSDataset(path.join(davis_root, 'JPEGImages', '480p'),
                        path.join(davis_root, 'Annotations', '480p'), max_skip, is_bl=False, subset=load_sub_davis())
    train_dataset = ConcatDataset([davis_dataset]*5 + [yv_dataset])

    print('YouTube dataset size: ', len(yv_dataset))
    print('DAVIS dataset size: ', len(davis_dataset))
    print('Concat dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)

def renew_bl_loader(max_skip):
    train_dataset = VOSDataset(path.join(bl_root, 'JPEGImages'),
                        path.join(bl_root, 'Annotations'), max_skip, is_bl=True)

    print('Blender dataset size: ', len(train_dataset))
    print('Renewed with skip: ', max_skip)

    return construct_loader(train_dataset)

"""
Dataset related
"""

"""
These define the training schedule of the distance between frames
We will switch to skip_values[i] once we pass the percentage specified by increase_skip_fraction[i]
Not effective for stage 0 training
"""
skip_values = [10, 15, 20, 25, 5]

if para['stage'] == 0:
    static_root = path.expanduser(para['static_root'])
    fss_dataset = StaticTransformDataset(path.join(static_root, 'fss'), method=0)
    duts_tr_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TR'), method=1)
    duts_te_dataset = StaticTransformDataset(path.join(static_root, 'DUTS-TE'), method=1)
    ecssd_dataset = StaticTransformDataset(path.join(static_root, 'ecssd'), method=1)

    big_dataset = StaticTransformDataset(path.join(static_root, 'BIG_small'), method=1)
    hrsod_dataset = StaticTransformDataset(path.join(static_root, 'HRSOD_small'), method=1)

    # BIG and HRSOD have higher quality, use them more
    train_dataset = ConcatDataset([fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset]
             + [big_dataset, hrsod_dataset]*5)
    train_sampler, train_loader = construct_loader(train_dataset)

    print('Static dataset size: ', len(train_dataset))

elif para['stage'] == 1:
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.8, 1.0]
    bl_root = path.join(path.expanduser(para['bl_root']))

    train_sampler, train_loader = renew_bl_loader(5)
    renew_loader = renew_bl_loader

else:
    # stage 2 or 3
    increase_skip_fraction = [0.1, 0.2, 0.3, 0.4, 0.9, 1.0]
    # VOS dataset, 480p is used for both datasets
    yv_root = path.join(path.expanduser(para['yv_root']), 'train_480p')
    davis_root = path.join(path.expanduser(para['davis_root']), '2017', 'trainval')

    train_sampler, train_loader = renew_vos_loader(5)
    renew_loader = renew_vos_loader


"""
Determine current/max epoch
"""
total_epoch = math.ceil(para['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
if para['stage'] != 0:
    increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
    # Skip will only change after an epoch, not in the middle
    print('The skip value will increase approximately at the following epochs: ', increase_skip_epoch[:-1])

"""
Starts training
"""
# Need this to select random bases in different workers
np.random.seed(np.random.randint(2**30-1) + local_rank*100)
try:
    for e in range(current_epoch, total_epoch):
        print('Epoch %d/%d' % (e, total_epoch))
        if para['stage']!=0 and e!=total_epoch and e>=increase_skip_epoch[0]:
            while e >= increase_skip_epoch[0]:
                cur_skip = skip_values[0]
                skip_values = skip_values[1:]
                increase_skip_epoch = increase_skip_epoch[1:]
            print('Increasing skip to: ', cur_skip)
            train_sampler, train_loader = renew_loader(cur_skip)

        # Train loop
        model.train()
        for data in train_loader:
            model.do_pass(data, total_iter)
            total_iter += 1

            if total_iter >= para['iterations']:
                break

finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter)

