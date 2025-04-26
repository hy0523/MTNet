from .convnextv2 import convnextv2_nano
import timm
import torch


def build_convnextv2_model(model_type, freeze_at=0):
    if model_type == 'convnextv2_nano.fcmae_ft_in22k_in1k':
        model = convnextv2_nano(drop_path_rate=0.4,
                                head_init_scale=1.0)
        checkpoint = torch.load('pretrain_models/convnextv2_nano_22k_224_ema.pt')
        model.load_state_dict(checkpoint["model"], strict=False)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
