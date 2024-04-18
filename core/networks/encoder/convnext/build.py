from .convnext import convnext_tiny, convnext_small
import timm


def build_convnext_model(model_type, freeze_at=0):

    if model_type == 'convnext-tiny':
        model = convnext_tiny()

    elif model_type == 'convnext-small':
        model = convnext_small()

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
