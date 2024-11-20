def create_clip_vit_L(img_size=224, use_checkpoint=False, precision='fp16'):
    model = VisionTransformer(input_resolution=img_size, patch_size=14,
        width=1024, layers=23, heads=16, use_grad_checkpointing=use_checkpoint)
    url = (
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/clip_vit_L.pth'
        )
    cached_file = download_cached_file(url, check_hash=False, progress=True)
    state_dict = torch.load(cached_file, map_location='cpu')
    interpolate_pos_embed(model, state_dict)
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    if precision == 'fp16':
        convert_weights_to_fp16(model)
    return model
