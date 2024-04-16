import math


def adjust_learning_rate(optimizer,
                         base_lr,
                         p,
                         itr,
                         max_itr,
                         restart=1,
                         warm_up_steps=1000,
                         warmup_lr_start=1e-6,
                         is_cosine_decay=False,
                         min_lr=0.,
                         encoder_lr_ratio=1.0):
    if itr < warm_up_steps:
        now_lr = warmup_lr_start + (base_lr - warmup_lr_start) * itr / warm_up_steps
    else:
        itr = itr - warm_up_steps
        max_itr = max_itr - warm_up_steps
        if is_cosine_decay:
            now_lr = min_lr + (base_lr - min_lr) * (math.cos(math.pi * itr /
                                                             (max_itr + 1)) +
                                                    1.) * 0.5
        else:
            now_lr = min_lr + (base_lr - min_lr) * (1 - itr / (max_itr + 1)) ** p

    # optimizer.param_groups[0]['lr'] = (now_lr - min_lr) * encoder_lr_ratio + min_lr
    optimizer.param_groups[0]['lr'] = now_lr
    optimizer.param_groups[1]['lr'] = now_lr
    # optimizer.param_groups['lr'] = now_lr
    return now_lr


def get_trainable_params(model,
                         base_lr,
                         weight_decay,
                         use_frozen_bn=False,
                         exclusive_wd_dict={},
                         no_wd_keys=[]):
    params = []
    memo = set()
    total_param = 0
    for key, value in model.named_parameters():
        if value in memo:
            continue
        total_param += value.numel()
        if not value.requires_grad:
            continue
        memo.add(value)
        wd = weight_decay
        for exclusive_key in exclusive_wd_dict.keys():
            if exclusive_key in key:
                wd = exclusive_wd_dict[exclusive_key]
                break
        if len(value.shape) == 1:  # normalization layers
            if 'bias' in key:  # bias requires no weight decay
                wd = 0.
            elif not use_frozen_bn:  # if not use frozen BN, apply zero weight decay
                wd = 0.
            elif 'encoder.' not in key:  # if use frozen BN, apply weight decay to all frozen BNs in the encoder
                wd = 0.
        else:
            for no_wd_key in no_wd_keys:
                if no_wd_key in key:
                    wd = 0.
                    break
        params += [{
            "params": [value],
            "lr": base_lr,
            "weight_decay": wd,
            "name": key
        }]

    print('Total Param: {:.2f}M'.format(total_param / 1e6))
    return params


def freeze_params(module):
    for p in module.parameters():
        p.requires_grad = False
