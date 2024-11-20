@register_model
def vit_base_patch16_reg8_gap_256(pretrained: bool=False, **kwargs
    ) ->VisionTransformer:
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12,
        class_token=False, no_embed_class=True, global_pool='avg', reg_tokens=8
        )
    model = _create_vision_transformer('vit_base_patch16_reg8_gap_256',
        pretrained=pretrained, **dict(model_args, **kwargs))
    return model
