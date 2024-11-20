def load_pretrained_imagenet(model, pretrained_model, cfg=None,
    ignore_classifier=True, num_frames=8, num_patches=196, **kwargs):
    import timm
    logging.info(f'Loading vit_base_patch16_224 checkpoints.')
    loaded_state_dict = timm.models.vision_transformer.vit_base_patch16_224(
        pretrained=True).state_dict()
    del loaded_state_dict['head.weight']
    del loaded_state_dict['head.bias']
    new_state_dict = loaded_state_dict.copy()
    for key in loaded_state_dict:
        if 'blocks' in key and 'attn' in key:
            new_key = key.replace('attn', 'temporal_attn')
            if not new_key in loaded_state_dict:
                new_state_dict[new_key] = loaded_state_dict[key]
            else:
                new_state_dict[new_key] = loaded_state_dict[new_key]
        if 'blocks' in key and 'norm1' in key:
            new_key = key.replace('norm1', 'temporal_norm1')
            if not new_key in loaded_state_dict:
                new_state_dict[new_key] = loaded_state_dict[key]
            else:
                new_state_dict[new_key] = loaded_state_dict[new_key]
    loaded_state_dict = new_state_dict
    loaded_keys = loaded_state_dict.keys()
    model_keys = model.state_dict().keys()
    load_not_in_model = [k for k in loaded_keys if k not in model_keys]
    model_not_in_load = [k for k in model_keys if k not in loaded_keys]
    toload = dict()
    mismatched_shape_keys = []
    for k in model_keys:
        if k in loaded_keys:
            if model.state_dict()[k].shape != loaded_state_dict[k].shape:
                mismatched_shape_keys.append(k)
            else:
                toload[k] = loaded_state_dict[k]
    logging.info('Keys in loaded but not in model:')
    logging.info(
        f'In total {len(load_not_in_model)}, {sorted(load_not_in_model)}')
    logging.info('Keys in model but not in loaded:')
    logging.info(
        f'In total {len(model_not_in_load)}, {sorted(model_not_in_load)}')
    logging.info('Keys in model and loaded, but shape mismatched:')
    logging.info(
        f'In total {len(mismatched_shape_keys)}, {sorted(mismatched_shape_keys)}'
        )
    model.load_state_dict(toload, strict=False)
