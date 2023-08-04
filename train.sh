CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 python -m torch.distributed.launch --master_port 13000 --nproc_per_node=2 train.py --id 0 --stage 3
