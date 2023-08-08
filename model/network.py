"""
network.py - The core of the neural network
Defines the structure and memory operations
Modifed from STM: https://github.com/seoungwugoh/STM

The trailing number of a variable usually denote the stride
e.g. f16 -> encoded features with stride 16
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import *


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        """
        f16: query feature read from memory (N, 1024, H/16, W/16)
        f8:  query feature from key encoder on query frame (N, 512, H/8, W/8)
        f4:  same as f8: (N, 256, H/4, W/4)
        """
        x = self.compress(f16)  # (N, 512, H/16, W/16)
        x = self.up_16_8(f8, x) # (N, 256, H/8, W/8)
        x = self.up_8_4(f4, x)  # (N, 256, H/4, W/4)

        x = self.pred(F.relu(x))# (N, 1, H/4, W/4)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) # (N, 1, H, W) -> enlarge to mask size of 384, 384
        return x

class CBDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.compress = ResBlock(2048, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256) # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256) # 1/8 -> 1/4

        self.pred = nn.Conv2d(256, 1, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, f16, f8, f4):
        """
        f16: query feature read from memory (N, 1024, H/16, W/16)
        f8:  query feature from key encoder on query frame (N, 512, H/8, W/8)
        f4:  same as f8: (N, 256, H/4, W/4)
        """
        x = self.compress(f16)  # (N, 512, H/16, W/16)
        x = self.up_16_8(f8, x) # (N, 256, H/8, W/8)
        x = self.up_8_4(f4, x)  # (N, 256, H/4, W/4)

        x = self.pred(F.relu(x))# (N, 1, H/4, W/4)

        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) # (N, 1, H, W) -> enlarge to mask size of 384, 384
        return x


class MemoryReader(nn.Module):
    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk):
        """
        mk := memory key: (N, 64, 1, H/16, W/16)
                -> Can be extend to (N, 64, T, H/16, W/16) for larger memory
        qk := query key : (N, 64, H/16, W/16)
        """
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2) # (N, 64 * T, H/16*W/16)
        qk = qk.flatten(start_dim=2) # (N, 64, H/16*W/16)

        # See supplementary material
        a_sq = mk.pow(2).sum(1).unsqueeze(2)
        ab = mk.transpose(1, 2) @ qk

        affinity = (2*ab-a_sq) / math.sqrt(CK)   # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity # (N, THW, HW), note that H <- H/16, W <- W/16

    def readout(self, affinity, mv, qv):
        """
        affinity: (N, T * HW, HW): pixel-wise scores from query to all frames in memory
        mv := memory value: concatenate of feature from value encoder on each (frame, mask)
                (8, 512, T, 24, 24)
        qv := query value but actually f16_thin from query key
                (8, 512, 24, 24)
        """
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T*H*W)  # (N, 512, T * HW)
        mem = torch.bmm(mo, affinity) # (N, 512, T * HW) @ (N, T * HW, HW) -> (N, 512, HW)
        mem = mem.view(B, CV, H, W) # (N, 512, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out # (N, 1024, H, W) with H <- H/16, W <- W/16


class STCN(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        """
        Soft-aggregation to combine multiple probability masks. Ideas from STM paper.
        prob: probability of mask for multiple objects (N, num_objects, H, W)
        Note: in the STM paper, p_i,m = softmax(logit(pred_p_i,m)) but seems like they didn't use
        the true softmax = exp(x) / sum(exp(x)) but instead use exp(x) = x / (1-x).
        """
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True), # prod of probility of not being object, (N, 1, H, W)
            prob # (N, 2, H, W)
        ], 1).clamp(1e-7, 1-1e-7) # (N, 3, H, W)
        new_prob = new_prob / (1 - new_prob) # demonimator of Eq (1) in STM supplementary.
        logits = torch.log(new_prob)  # Using log to avoid numerical instability.
        # logits = torch.log((new_prob /(1-new_prob))) # original code
        return logits # (N, 3, H, W)

    def encode_key(self, frame):
        """
        Encode frames into key space.
        Input: frame -> (N, T, 3, H, W)
        During training, T = 3
        """
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1)) # input: (N*T, C, H, W)
        # f16: (N*T, 1024, H/16, W/16)
        # f8: (N*T, 512, H/8, W/8)
        # f4: (N*T, 256, H/4, W/4)
        k16 = self.key_proj(f16)        # (N*T, 64, H/16, W/16)
        f16_thin = self.key_comp(f16)   # (N*T, 512, H/16, W/16)

        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous() # (N, 64, T, H/16, W/16)
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:]) # (N, T, 512, H/16, W/16)
        f16 = f16.view(b, t, *f16.shape[-3:])   # (N, T, 1024, H/16, W/16)
        f8 = f8.view(b, t, *f8.shape[-3:])      # (N, T, 512, H/8, W/8)
        f4 = f4.view(b, t, *f4.shape[-3:])      # (N, T, 256, H/4, W/4)

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None):
        """
        Training:
            Extract memory key/value for a frame and binary mask of an object,
                other_mask is binary mask of second object.
            frame:      (N, C=3, H, W)
            kf16:       (N, 1024, H/16, W/16)
            mask:       (N, C=1, H, W)
            other_mask: (N, C=1, H, W)
        Inference: TODO
        """
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask) # (N, 512, H/16, W/16)

        return f16.unsqueeze(2) # (N, 512, T=1, H/16, W/16)

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None):
        # q - query, m - memory
        # qk16 := k16, query key                        (N, 64, 24, 24)
        # qv16 := f16_thin, query value                 (N, 512, 24, 24)
        # qf8 := f8, query feature 8                    (N, 512, 48, 48)
        # qf4 := f4, query feature 4                    (N, 256, 96, 96)
        # mk16 := memory key from previous frames       (N, 64, memory_len, 24, 24)
        # mv16 := memory value from previous frames     (N, num_object=2, 512, memory_len, 24, 24)

        # Compute affinity between f16 feature from RGB query and memory
        affinity = self.memory.get_affinity(mk16, qk16) # (N, memory_len*576, 576), 24 * 24 = 576

        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            logits = torch.cat([
                # Decode first memory object
                self.decoder(self.memory.readout(affinity,  # (N, memory_len*576, 576)
                                                 mv16[:, 0],# (N, 512, memory_len, 24, 24) <- 0 means first object
                                                 qv16),     # (N, 512, 24, 24) -> Output: (N, 1024, 24, 24)
                             qf8,   # (N, 512, 48, 48)
                             qf4    # (N, 256, 96, 96)
                             ), # (N, 1, H, W)
                self.decoder(self.memory.readout(affinity,
                                                 mv16[:,1],
                                                 qv16),
                             qf8,
                             qf4), # (N, 1, H, W)
            ], 1) # (N, 2, H, W)

            prob = torch.sigmoid(logits) # (N, 2, H, W)
            # selector (in training, only 2 selected objects) -> (N, 2) -> (N, 2, 1, 1) after double unsqueeze
            # cancel out the probability of not selected object (e.g sequence only has 1 object)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob) # (N, 3, H, W)
        prob = F.softmax(logits, dim=1)[:, 1:] # (N, 2, H, W), ignore channel (1-prob) * (1-prob)

        return logits, prob

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError

class STCNCB(nn.Module):
    def __init__(self, single_object):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()
        self.cb_decoder = CBDecoder()

    def aggregate(self, prob):
        """
        Soft-aggregation to combine multiple probability masks. Ideas from STM paper.
        prob: probability of mask for multiple objects (N, num_objects, H, W)
        Note: in the STM paper, p_i,m = softmax(logit(pred_p_i,m)) but seems like they didn't use
        the true softmax = exp(x) / sum(exp(x)) but instead use exp(x) = x / (1-x).
        """
        new_prob = torch.cat([
            torch.prod(1-prob, dim=1, keepdim=True), # prod of probility of not being object, (N, 1, H, W)
            prob # (N, 2, H, W)
        ], 1).clamp(1e-7, 1-1e-7) # (N, 3, H, W)
        new_prob = new_prob / (1 - new_prob) # demonimator of Eq (1) in STM supplementary.
        logits = torch.log(new_prob)  # Using log to avoid numerical instability.
        # logits = torch.log((new_prob /(1-new_prob))) # original code
        return logits # (N, 3, H, W)

    def encode_key(self, frame):
        """
        Encode frames into key space.
        Input: frame -> (N, T, 3, H, W)
        During training, T = 3
        """
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1)) # input: (N*T, C, H, W)
        # f16: (N*T, 1024, H/16, W/16)
        # f8: (N*T, 512, H/8, W/8)
        # f4: (N*T, 256, H/4, W/4)
        k16 = self.key_proj(f16)        # (N*T, 64, H/16, W/16)
        f16_thin = self.key_comp(f16)   # (N*T, 512, H/16, W/16)

        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous() # (N, 64, T, H/16, W/16)
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:]) # (N, T, 512, H/16, W/16)
        f16 = f16.view(b, t, *f16.shape[-3:])   # (N, T, 1024, H/16, W/16)
        f8 = f8.view(b, t, *f8.shape[-3:])      # (N, T, 512, H/8, W/8)
        f4 = f4.view(b, t, *f4.shape[-3:])      # (N, T, 256, H/4, W/4)

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None):
        """
        Training:
            Extract memory key/value for a frame and binary mask of an object,
                other_mask is binary mask of second object.
            frame:      (N, C=3, H, W)
            kf16:       (N, 1024, H/16, W/16)
            mask:       (N, C=1, H, W)
            other_mask: (N, C=1, H, W)
        Inference: TODO
        """
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask) # (N, 512, H/16, W/16)

        return f16.unsqueeze(2) # (N, 512, T=1, H/16, W/16)

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None):
        # q - query, m - memory
        # qk16 := k16, query key                        (N, 64, 24, 24)
        # qv16 := f16_thin, query value                 (N, 512, 24, 24)
        # qf8 := f8, query feature 8                    (N, 512, 48, 48)
        # qf4 := f4, query feature 4                    (N, 256, 96, 96)
        # mk16 := memory key from previous frames       (N, 64, memory_len, 24, 24)
        # mv16 := memory value from previous frames     (N, num_object=2, 512, memory_len, 24, 24)

        # Compute affinity between f16 feature from RGB query and memory
        affinity = self.memory.get_affinity(mk16, qk16) # (N, memory_len*576, 576), 24 * 24 = 576

        if self.single_object:
            logits = self.decoder(self.memory.readout(affinity, mv16, qv16), qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            read_first_mem = self.memory.readout(affinity,  # (N, memory_len*576, 576)
                                                 mv16[:, 0],# (N, 512, memory_len, 24, 24) <- 0 means first object
                                                 qv16)     # (N, 512, 24, 24) -> Output: (N, 1024, 24, 24)

            read_second_mem = self.memory.readout(affinity,
                                                mv16[:,1],
                                                qv16)

            # Decode mask first
            logits = torch.cat([
                # Decode first memory object
                self.decoder(read_first_mem,
                             qf8,   # (N, 512, 48, 48)
                             qf4    # (N, 256, 96, 96)
                             ), # (N, 1, H, W)
                self.decoder(read_second_mem,
                             qf8,
                             qf4), # (N, 1, H, W)
            ], 1) # (N, 2, H, W)

            # Decode contact boundary
            cb_logits = self.cb_decoder(torch.cat([read_first_mem, read_second_mem], 1),
                                        qf8,
                                        qf4)

            prob = torch.sigmoid(logits) # (N, 2, H, W)
            # selector (in training, only 2 selected objects) -> (N, 2) -> (N, 2, 1, 1) after double unsqueeze
            # cancel out the probability of not selected object (e.g sequence only has 1 object)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob) # (N, 3, H, W)
        prob = F.softmax(logits, dim=1)[:, 1:] # (N, 2, H, W), ignore channel (1-prob) * (1-prob)

        return logits, prob, cb_logits

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        else:
            raise NotImplementedError
