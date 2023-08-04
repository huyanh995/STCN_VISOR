# Note on all the datasets output

## STATIC
'rgb'       --- (3, 3, 384, 384)  -> torch.Tensor
'gt': masks --- (3, 1, 384, 384)  -> torch.Tensor
'cls_gt'    --- (3, 384, 384)     -> np.ndarray
'info'



## VOS (YTVOS, DAVIS, etc)
```
'rgb'       --- (N, 3, 384, 384)    -> torch.tensor
'gt'        --- (N, 1, 384, 384)    -> np.array
'cls_gt'    --- (N, 384, 384)       -> np.array
'sec_gt'    --- (N, 1, 384, 384)    -> np.array
'selector'  --- (4, )               -> torch.tensor
'info'
```

## VISOR
```
'rgb'       --- (N, 3, 384, 384)    -> torch.tensor
'gt'        --- (N, 1, 384, 384)    -> np.array
'cls_gt'    --- (N, 384, 384)       -> np.array
'sec_gt'    --- (N, 1, 384, 384)    -> np.array
'left_hand' --- (N, 1, 384, 384)    -> np.array
'right_hand'--- (N, 1, 384, 384)    -> np.array
'boundary'  --- (N, 2, 384, 384)    -> np.array
'selector'  --- (4, )               -> torch.tensor
'info'
```

## EgoHOS
```
'rgb'       --- (N=3, 3, 384, 384)    -> torch.tensor
'gt'        --- (N=3, 1, 384, 384)    -> np.array
'cls_gt'    --- (N=3, 384, 384)       -> np.array
'sec_gt'    --- (N=3, 1, 384, 384)    -> np.array
'left_hand' --- (N=3, 1, 384, 384)    -> np.array
'right_hand'--- (N=3, 1, 384, 384)    -> np.array
'boundary'  --- (N=3, 2, 384, 384)    -> np.array
'selector'  --- (4, )                 -> torch.tensor
'info'
```

## DAVIS TEST
```
'rgb'       --- (num_frames, 3, H, W)
'gt'        --- (num_labels, num_frames, 1, H, W)
'info':
```

## YTVOS TEST
```
'rgb'       --- (n_frames, 3, H, W) -> torch.Tensor
'gt'        --- (n_labels, n_frames, 1, H, W) -> torch.Tensor
'info'      --- has labels, label mappings inside
```

## VISOR TEST
Should be the same as YTVOS.
```
'rgb'
'gt'
'info'
```
