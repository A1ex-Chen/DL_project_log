def text2img(*, args, checkpoint_map_location):
    print('loading text2img')
    text2img_checkpoint = torch.load(args.text2img_checkpoint_path,
        map_location=checkpoint_map_location)
    unet_model = unet_model_from_original_config()
    unet_diffusers_checkpoint = (
        unet_original_checkpoint_to_diffusers_checkpoint(unet_model,
        text2img_checkpoint))
    del text2img_checkpoint
    load_checkpoint_to_model(unet_diffusers_checkpoint, unet_model, strict=True
        )
    print('done loading text2img')
    return unet_model
