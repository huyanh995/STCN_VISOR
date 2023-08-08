"""
Training code for VISOR and EgoHOS dataset.
EgoHOS is a static dataset.
"""

import os
import math
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

# Import model and dataset
from model.model_visor import STCNVISORModel
from dataset.egohos_dataset import EgoHOSDataset
from dataset.visor_dataset import VISORDataset

from util.hyper_para import HyperParameters


#* Initial setup ====================================================


# Set seed to ensure the same initialization
torch.manual_seed(14159265)
np.random.seed(14159265)
random.seed(14159265)

# Parse command line arguments
para = HyperParameters()
para.parse()

os.environ['CUDA_VISIBLE_DEVICES'] = para['gpu_id']
print(f'Using GPU {para["gpu_id"]} ...')

# Helper functions
def construct_loader(dataset):
    train_loader = DataLoader(dataset,
                              para['batch_size'],
                              num_workers = para['num_workers'],
                              drop_last = True,
                              pin_memory = True) # In single GPU doesn't need worker_init_fn
    return train_loader

def renew_visor(skip_frame = False):
        visor_root = os.path.expanduser(para['visor_root'])
        visor_dataset = VISORDataset(im_root = os.path.join(visor_root, 'JPEGImages'),
                                    gt_root = os.path.join(visor_root, 'Annotations'),
                                    bound_root = os.path.join(visor_root, 'Boundaries'),
                                    skip_frame = skip_frame,
                                    include_hand = para['include_hand'])

        return construct_loader(visor_dataset)

#* Model preparation ================================================
model = STCNVISORModel(para).train()
if para['load_model'] is not None:
    total_iter = model.load_model(para['load_model'])
    print('Previously trained model loaded!')
else:
    total_iter = 0

if para['load_network'] is not None:
    model.load_network(para['load_network'])
    print('Previously trained network loaded!')

#* Dataset preparation ==============================================
# Construct dataset
def construct_visor(skip: False):
    pass

if para['stage'] == 0:
    # EgoHOS Static dataset
    egohos_dataset = EgoHOSDataset(os.path.expanduser(para['ego_root']),
                                   subset='train',
                                   prob=0.2)
    train_loader = construct_loader(egohos_dataset)
    pass

elif para['stage'] == 1:
    # VISOR dataset
    train_loader = renew_visor(skip_frame = False) # default not skip frame

#* Prepare training =================================================
total_epoch = math.ceil(para['iterations']/len(train_loader))
current_epoch = total_iter // len(train_loader)
print('Number of training epochs (the last epoch might not complete): ', total_epoch)
if para['stage'] == 1:
    # VISOR training, begin skip frame at one third of the training
    skip_epoch = total_epoch // 3
    pass
#* Training code ====================================================
try:
    for e in range(current_epoch, total_epoch):
        print(f'Epoch {e}/{total_epoch} ...')
        if para['stage'] == 1 and e >= skip_epoch:
            train_loader = renew_visor(skip_frame=True)

        start = time.time()
        #! Train loop
        model.train()
        for data in train_loader:
            # weight = max(0.2, 0.5 * (1 - total_iter / para['iterations'])) # linearly decrease weight
            weight = -1
            model.do_pass(data, total_iter, weight)
            total_iter += 1
            print(f'One iteration takes {round(time.time() - start, 2)} seconds.')
            if total_iter >= para['iterations']:
                break

            if total_iter % 1000 == 0 and total_iter !=0:
                model.save(total_iter)

        if para['stage'] == 1 and e > skip_epoch:
            train_loader = renew_visor(skip_frame=True)

        print(f'Finished in {round((time.time() - start) / 60.0, 2)} minutes.')
        print('-' * 80)

finally:
    if not para['debug'] and model.logger is not None and total_iter>5000:
        model.save(total_iter)
