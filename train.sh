conda activate zyl
cd /home/zyl/code_dir/

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
--nproc_per_node=4 code/main.py --batch_size=2