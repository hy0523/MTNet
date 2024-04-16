from .convnext.build import build_convnext_model



def build_encoder(name):
    if 'convnext' in name:
        return build_convnext_model(model_type=name)

    else:
        raise NotImplementedError
