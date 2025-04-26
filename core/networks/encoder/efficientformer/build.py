from .efficientformer import efficientformer_l1_feat, efficientformer_l3_feat, efficientformer_l7_feat
import torch
from IPython import embed


def build_efficientformer_model(model_type, freeze_at=0):
    if model_type == 'efficientformer-l1':
        model = efficientformer_l1_feat(pretrained=False)
        checkpoint = torch.load('pretrain_models/efficientformer_l1_300d.pth')
        model.load_state_dict(checkpoint["model"], strict=False)

    elif model_type == 'efficientformer-l3':
        model = efficientformer_l3_feat(pretrained=False)
        checkpoint = torch.load('pretrain_models/efficientformer_l3_300d.pth')
        model.load_state_dict(checkpoint["model"], strict=False)

    elif model_type == 'efficientformer-l7':
        model = efficientformer_l7_feat(pretrained=False)
        checkpoint = torch.load('pretrain_models/efficientformer_l7_300d.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
