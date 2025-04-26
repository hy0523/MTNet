#!/bin/sh

GPU_ID=0,1,2,3,4,5,6,7
GPU_NUM="$(expr \( \( ${#GPU_ID} + 1 \) / 2 \))"
exp="exp"
model_name="MTNet"
encoder_name="convnext-tiny"
imsize=512
train_frames=3
clip_length=3

## Pre-train
stage="pre_train"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=9527 --use_env train.py \
--exp ${exp} \
--stage ${stage} \
--model_name ${model_name} \
--imsize ${imsize}  \
--encoder_name ${encoder_name}  \
--train_frames ${train_frames} \
--clip_length ${clip_length}

# Main-train
GPU_ID=0,1,2,3 #0,1,2,3
GPU_NUM="$(expr \( \( ${#GPU_ID} + 1 \) / 2 \))"

stage="main_train"
pretrain_model_path="${exp}/${model_name}/pre_train/best.pth"
CUDA_VISIBLE_DEVICES=${GPU_ID} python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port=9527  --use_env train.py \
--exp ${exp} \
--stage ${stage} \
--model_name ${model_name} \
--imsize ${imsize} \
--encoder_name ${encoder_name} \
--train_frames ${train_frames} \
--clip_length ${clip_length} \
--pretrain_model_path ${pretrain_model_path} \
--max_iter ${max_iter} \
--eval_iter ${eval_iter}
