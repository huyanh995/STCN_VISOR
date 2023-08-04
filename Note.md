# Personal notes about implementation of the project

## Model
Structure:



Input: video sequence and first-frame mask -> Auto handle new objects in the middle of sequences.

## Dataloader
- There are 2 dataloaders for each dataset, a general one is in vos_dataset.py

### Training
- Load 3 pairs (image, mask) from a video sequence -> ensure different length sequence can load into a batch.
- Pairs are randomly picked, but controlled via max_jump (distance), to not favor the few first frames.
- Apply transformations randomly to the pairs -> data augmentation.
- For 3 pairs, randomly pick 2 objects to be target and second target.
- RGB Images: (N, T, 3, H, W)
- gt: (N, T, 1, H, W) -> 1 mean only select target object
- sec_gt: (N, T, 1, H, W) -> for second target object, if no, all 0s.
- cls_gt: combine sec_gt and gt into a single mask.

### Validating/Testing


## Encoders
Terminology:
- Query: new RGB image in the sequence
- Key: feature exacted from RGB in the sequence
- Value: feature exacted from RGB and mask in the sequence

### Key encoder (KeyEncoder class in modules.py)
- Note: there is difference between training and inference (as stated in DataLoader above)
    - In training: data is a batch N sequences, each sequence has 3 frames and 2 selected objects (randomly).
    - In inference: due to varied sequence length, data is a batch of 1 sequence, with all frames and all objects. # TODO: check this
- Encode every RGB frame in the sequence, using ResNet50
- Each RGB frame is only encoded once.
- Input: (N, T, 3, H, W) tensors
- Output: Immediate feature vectors f4:     (N, T, 256, H/4, W/4)
                                    f8:     (N, T, 512, H/8, W/8)
                                    f16:    (N, T, 1024, H/16, W/16) <- final feature map

- There are two additional feats    k16:    (N, 64, T, H/16, W/16) <- projection of f16 through Conv2d to reduce # channels
                                    f16_thin: (N, T, 512, H/16, W/16), same as k16 but with 1024 -> 512 channels

- How keys are storing in memory

### Value encoder
- Produce value feature from RGB, its key feature and mask(s).
- Input:    RGB images:     (N, T, 3, H, W)
            f16 RGB feat:   (N, T, 1024, H/16, W/16) -> kf16
            mask:           (N, T, 1, H, W) # binary mask for 1st object
            other_masks:    (N, T, 1, H, W) # binary mask for 2nd object

- Concatenate RGB, mask, other_mask -> a Tensor (N, T, 5, H, W)
    -> feed through *modified* ResNet18
    -> (N, 256, H/16, W/16) feature.
- Using FeatureFusionBlock (using attention mechanism) to fuse above feature
    with f16 key feature from RGB frame
- Output:   (N, 512, H/16, W/16) value feature.

## Decoder
### Affinity
- Using only f16 feature from RGB query and RGB memory frames to negative L2 distance (details in paper).
    and softmax to get probability.
- Memory is (N, 64 * T, H, W), query is (N, 64, H, W), where T is number of previous frames
    Affinity -> (N, T * HW, HW) -> Pixel-wise affinity matrix.

### Memory reader
- Training: i.e 3 pairs of (frame, mask), with at most 2 objects per sequence being selected.
    -



### Decoder
- Using Upsampling block
