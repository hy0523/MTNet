#!/bin/sh

python main.py \
	--method 'MTNet' \
	--dataset 'DAVIS16' \
	--gt_dir ./gt/ \
	--pred_dir ./result/ \
