#!/bin/sh

exp="exp"
model_name="MTNet"
encoder_name="convnext-tiny"
imsize=512
train_frames=3
clip_length=3

# Pre-train
stage="pre_train"
python  train.py \
--exp ${exp} \
--stage ${stage} \
--model_name ${model_name} \
--imsize ${imsize} \
--encoder_name ${encoder_name}

# Main-train
stage="main_train"
pretrain_model_path="${exp}/${model_name}/pre_train/best.pth"
python  train.py \
--exp ${exp} \
--stage ${stage} \
--model_name ${model_name} \
--imsize ${imsize} \
--encoder_name ${encoder_name} \
--pretrain_model_path ${pretrain_model_path}
