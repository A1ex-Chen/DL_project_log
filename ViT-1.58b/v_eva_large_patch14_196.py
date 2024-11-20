@register_model
def eva_large_patch14_196(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    """ EVA-large model https://arxiv.org/abs/2211.07636 /via MAE MIM pretrain"""
    model_args = dict(patch_size=14, embed_dim=1024, depth=24, num_heads=16,
        global_pool='avg')
    model = _create_vision_transformer('eva_large_patch14_196', pretrained=
        pretrained, **dict(model_args, **kwargs))
    return model
