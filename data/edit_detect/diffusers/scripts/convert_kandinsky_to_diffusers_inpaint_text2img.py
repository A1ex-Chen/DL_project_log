def inpaint_text2img(*, args, checkpoint_map_location):
    print('loading inpaint text2img')
    inpaint_text2img_checkpoint = torch.load(args.
        inpaint_text2img_checkpoint_path, map_location=checkpoint_map_location)
    inpaint_unet_model = inpaint_unet_model_from_original_config()
    inpaint_unet_diffusers_checkpoint = (
        inpaint_unet_original_checkpoint_to_diffusers_checkpoint(
        inpaint_unet_model, inpaint_text2img_checkpoint))
    del inpaint_text2img_checkpoint
    load_checkpoint_to_model(inpaint_unet_diffusers_checkpoint,
        inpaint_unet_model, strict=True)
    print('done loading inpaint text2img')
    return inpaint_unet_model
