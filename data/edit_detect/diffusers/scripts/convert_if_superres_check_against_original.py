def superres_check_against_original(dump_path, unet_checkpoint_path):
    model_path = dump_path
    model = UNet2DConditionModel.from_pretrained(model_path)
    model.to('cuda')
    orig_path = unet_checkpoint_path
    if '-II-' in orig_path:
        from deepfloyd_if.modules import IFStageII
        if_II_model = IFStageII(device='cuda', dir_or_name=orig_path,
            model_kwargs={'precision': 'fp32'}).model
    elif '-III-' in orig_path:
        from deepfloyd_if.modules import IFStageIII
        if_II_model = IFStageIII(device='cuda', dir_or_name=orig_path,
            model_kwargs={'precision': 'fp32'}).model
    batch_size = 1
    channels = model.config.in_channels // 2
    height = model.config.sample_size
    width = model.config.sample_size
    height = 1024
    width = 1024
    torch.manual_seed(0)
    latents = torch.randn((batch_size, channels, height, width), device=
        model.device)
    image_small = torch.randn((batch_size, channels, height // 4, width // 
        4), device=model.device)
    interpolate_antialias = {}
    if 'antialias' in inspect.signature(F.interpolate).parameters:
        interpolate_antialias['antialias'] = True
        image_upscaled = F.interpolate(image_small, size=[height, width],
            mode='bicubic', align_corners=False, **interpolate_antialias)
    latent_model_input = torch.cat([latents, image_upscaled], dim=1).to(model
        .dtype)
    t = torch.tensor([5], device=model.device).to(model.dtype)
    seq_len = 64
    encoder_hidden_states = torch.randn((batch_size, seq_len, model.config.
        encoder_hid_dim), device=model.device).to(model.dtype)
    fake_class_labels = torch.tensor([t], device=model.device).to(model.dtype)
    with torch.no_grad():
        out = if_II_model(latent_model_input, t, aug_steps=
            fake_class_labels, text_emb=encoder_hidden_states)
    if_II_model.to('cpu')
    del if_II_model
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    print(50 * '=')
    with torch.no_grad():
        noise_pred = model(sample=latent_model_input, encoder_hidden_states
            =encoder_hidden_states, class_labels=fake_class_labels, timestep=t
            ).sample
    print('Out shape', noise_pred.shape)
    print('Diff', (out - noise_pred).abs().sum())
