def checkpoint_filter_fn(state_dict: Dict[str, torch.Tensor], model:
    VisionTransformer, adapt_layer_scale: bool=False, interpolation: str=
    'bicubic', antialias: bool=True) ->Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''
    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')
    if 'mask_token' in state_dict:
        state_dict = _convert_dinov2(state_dict, model)
    if 'encoder' in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'
    if 'visual.trunk.pos_embed' in state_dict:
        prefix = 'visual.trunk.'
    if prefix:
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if
            k.startswith(prefix)}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(v, (H, W), interpolation=
                    interpolation, antialias=antialias, verbose=True)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False
                ) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(v, new_size=model.patch_embed.
                grid_size, num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation, antialias=antialias, verbose=True)
        elif adapt_layer_scale and 'gamma_' in k:
            k = re.sub('gamma_([0-9])', 'ls\\1.gamma', k)
        elif 'pre_logits' in k:
            continue
        out_dict[k] = v
    return out_dict
