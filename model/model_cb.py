import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from model.network import STCN, STCNCB
from model.losses import LossComputer, iou_hooks_mo, iou_hooks_so, CombineLossComputer, CBCombineLossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs

class STCNCBModel:
    """
    Only support single GPU yet
    """
    def __init__(self, para, save_path='./saves'):
        self.para = para
        self.logger = None
        self.loss = para['loss']
        # self.STCN = STCN(self.single_object).cuda()
        self.STCN = STCNCB(single_object=False).cuda() # VISOR and EgoHOS use multi-object

        self.save_path = os.path.join(save_path, self.para['name'])
        if os.path.exists(self.save_path):
            print('[WARNING] Save path already exists!')
        os.makedirs(self.save_path, exist_ok=True)

        self.loss_computer = CBCombineLossComputer()

        self.train()
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.STCN.parameters()),
                                    lr = para['lr'],
                                    weight_decay = 1e-7)

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones = para['steps'], # an increasing list of steps
                                                        gamma = para['gamma'])

        if para['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        self.log_loss = {'total_loss': 0.0,
                         'mask_loss': 0.0,
                         'boundary_loss': 0.0}

    def do_pass(self, data, it=0, weight=0.2):
        """
        VISOR and EgoHOS data format:
        'rgb'       --- (N=3, 3, 384, 384)    -> torch.tensor
        'gt'        --- (N=3, 1, 384, 384)    -> np.array
        'cls_gt'    --- (N=3, 384, 384)       -> np.array
        'sec_gt'    --- (N=3, 1, 384, 384)    -> np.array
        'left_hand' --- (N=3, 1, 384, 384)    -> np.array
        'right_hand'--- (N=3, 1, 384, 384)    -> np.array
        'boundary'  --- (N=3, 2, 384, 384)    -> np.array
        'selector'  --- (4, )                 -> torch.tensor
        'info'
        """
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        # Move data to GPU
        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        out = {}
        # Each data points are 3 x (frame, mask, boundary)
        frames = data['rgb'] # Load all rgb images: (N, 3, 3, 384, 384)
        masks = data['gt'] # Load all masks:        (N, 3, 1, 384, 384)
        boundaries = data['boundary_gt'] #          (N, 3, 2, 384, 384)

        # with torch.cuda.amp.autocast(enabled=self.para['amp']):
        with torch.cuda.amp.autocast(enabled=False):
            #*====== ENCODE KEYS =========================================================
            # key features never change, compute once
            k16, kf16_thin, kf16, kf8, kf4 = self.STCN('encode_key', frames)

            #*====== ENCODE VALUES =======================================================
            second_masks = data['sec_gt']
            selector = data['selector']

            #! Step 1: encode frame 0 with its first (ground truth) mask.
            ref_v1 = self.STCN('encode_value',
                               frames[:, 0],        # first frame rgb               (N, 3, 384, 384)
                               kf16[:, 0],          # first rgb feature             (N, 1024, 24, 24)
                               masks[:, 0],         # mask of first frame           (N, 1, 384, 384)
                               second_masks[:, 0]   # second mask of first frame    (N, 1, 384, 384)
                               ) # Output: (N, 512, T=1, H/16, W/16)

            # change the role of mask and second mask (other mask in function)
            ref_v2 = self.STCN('encode_value',
                               frames[:, 0],
                               kf16[:, 0],
                               second_masks[:, 0],  # change anchor mask order
                               masks[:, 0]
                               ) # Output: (N, 512, T=1, H/16, W/16)

            ref_v = torch.stack([ref_v1, ref_v2], 1) # (N, 2, 512, T=1, H/16, W/16)

            #*====== SEGMENT OBJECTS =====================================================
            #! Step 2a: segment frame 1 with frame 0 feature and encoded value
            # frame 1 -> query, frame 0 -> key
            prev_logits, prev_mask, prev_cb_logits = self.STCN('segment',
                                                    k16[:, :, 1],    # Query key feature 16:    #* (N, 64, 24, 24)
                                                    kf16_thin[:, 1], # Query value 16:          #* (N, 512, 24, 24)
                                                    kf8[:, 1],       # Query feature 8:         #* (N, 512, 48, 48)
                                                    kf4[:, 1],       # Query feature 4:         #* (N, 256, 96, 96)
                                                    k16[:, :, 0:1],  # memory key feature 16:   #* (N, 64, memory_len=1, 24, 24)
                                                    ref_v,           # memory value             #* (N, num_object=2, 512, memory_len=1, 24, 24)
                                                    selector)        # selector                 #* (N, num_object=2) or 4 in VISOR?

            # NOTE: Any reason to kf[:, :, 0:1] instead of kf[:, :, 0]? -> Same shape with kf16[:,:,1]
            # NOTE: prev_mask is not binary mask anymore, but a probability map.

            # Step 2b: Got (predicted) prev_mask of frame 1 -> encode frame 1 value
            prev_v1 = self.STCN('encode_value',
                                frames[:, 1],       # frame 1 rgb
                                kf16[:, 1],
                                prev_mask[:, 0:1],  # predicted mask of target object
                                prev_mask[:, 1:2]   # predicted mask of other object
                                ) # Output: (N, 512, 1, H/16, W/16)

            prev_v2 = self.STCN('encode_value',
                                frames[:,1],
                                kf16[:,1],
                                prev_mask[:, 1:2],
                                prev_mask[:, 0:1]
                                ) # (N, 512, 1, H/16, W/16)

            # Encoded values from previous (frame, predicted mask) pair.
            prev_v = torch.stack([prev_v1, prev_v2], 1) # (N, 2, 512, 1, H/16, W/16)
            # values from all previous (frame, mask) pairs.

            # Concatenate encoded values from frame 0 and frame 1, ready for segment frame 2
            values = torch.cat([ref_v, prev_v], 3) # (N, 2, 512, 2, H/16, W/16)

            del ref_v

            # Segment frame 2 with frame 0 and 1 concatenated encoded values
            this_logits, this_mask, this_cb_logits = self.STCN('segment',
                                                        k16[:, :, 2],         # third frame rgb
                                                        kf16_thin[:, 2],
                                                        kf8[:, 2],
                                                        kf4[:, 2],
                                                        k16[:, :, 0:2], # NOTE: Key idea: memory key is also from query key k16[:, :, 1] in previous step
                                                        values,
                                                        selector)

            #! Step 3: construct output and calculate loss, not counting frame 0 since GT is provided
            # depends on number of object, output consists of (mask_i, sec_mask_i, logits_i) for each object
            # these are after sigmoid activation
            out['mask_1'] = prev_mask[:, 0:1] # (N, 1, H, W)
            out['mask_2'] = this_mask[:, 0:1] # (N, 1, H, W)
            out['sec_mask_1'] = prev_mask[:, 1:2]
            out['sec_mask_2'] = this_mask[:, 1:2]
            out['cb_1'] = prev_cb_logits
            out['cb_2'] = this_cb_logits

            # NOTE: How to aggregate masks into one mask? Suppose a pixel can be 1 in both masks for 2 objects

            out['logits_1'] = prev_logits # (N, 3, H, W)
            out['logits_2'] = this_logits # (N, 3, H, W)

            if self._do_log or self._is_train:
                losses = self.loss_computer.compute({**data, **out}, it, weight)
                self.log(losses)

            # Backward pass
            # This should be done outside autocast
            # but I trained it like this and it worked fine
            # so I am keeping it this way for reference
            self.optimizer.zero_grad(set_to_none=True)

            if self.para['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                self.optimizer.step()
            self.scheduler.step()

        if it % self.para['log_every'] == 0 and it != 0:
            print(f'Iter {it} >>> losses {self.log_loss["total_loss"] / self.para["log_every"]}, mask loss {self.log_loss.get("mask_loss", 0.0)/ self.para["log_every"]}, boundary loss {self.log_loss.get("boundary_loss", 0)/ self.para["log_every"]}')
            self.reset_log()

    def save(self, it):
        now = datetime.now().strftime("%y_%m_%d_%H_%M")
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, f'{now}_iter_{it}.pth')
        torch.save(self.STCN.state_dict(), model_path)
        print('Model saved to %s.' % model_path)

        self.save_checkpoint(it)

    def save_checkpoint(self, it):
        checkpoint_path = os.path.join(self.save_path, 'checkpoint.pth')
        checkpoint = {
            'it': it,
            'network': self.STCN.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

        print('Checkpoint saved to %s.' % checkpoint_path)

    def load_model(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STCN.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        # map_location = 'cuda:%d' % self.local_rank
        # src_dict = torch.load(path, map_location={'cuda:0': map_location})

        src_dict = torch.load(path)

        # Maps SO weight (without other_mask) to MO weight (with other_mask)
        for k in list(src_dict.keys()):
            if k == 'value_encoder.conv1.weight':
                if src_dict[k].shape[1] == 4:
                    pads = torch.zeros((64,1,7,7), device=src_dict[k].device)
                    nn.init.orthogonal_(pads)
                    src_dict[k] = torch.cat([src_dict[k], pads], 1)

        # self.STCN.module.load_state_dict(src_dict)
        self.STCN.load_state_dict(src_dict, strict=False)
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        # Shall be in eval() mode to freeze BN parameters
        self.STCN.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.STCN.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STCN.eval()
        return self

    def log(self, losses):
        self.log_loss['total_loss'] += losses['total_loss'].item()
        if self.loss != 'default':
            self.log_loss['mask_loss'] += losses['mask_loss'].item()
            self.log_loss['boundary_loss'] += losses['boundary_loss'].item()

    def reset_log(self):
        with open(os.path.join(self.save_path, 'log.txt'), 'a') as f:
            f.write(f'{self.log_loss["total_loss"] / self.para["log_every"]}, {self.log_loss.get("mask_loss", 0.0)/ self.para["log_every"]}, {self.log_loss.get("boundary_loss", 0)/ self.para["log_every"]}\n')

        self.log_loss = {
            'total_loss': 0.0,
            'mask_loss': 0.0,
            'boundary_loss': 0.0}
