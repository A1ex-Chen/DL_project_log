def super_res_unet(*, args, checkpoint_map_location):
    print('loading super resolution unet')
    super_res_checkpoint = torch.load(args.super_res_unet_checkpoint_path,
        map_location=checkpoint_map_location)
    super_res_checkpoint = super_res_checkpoint['state_dict']
    super_res_first_model = (
        super_res_unet_first_steps_model_from_original_config())
    super_res_first_steps_checkpoint = (
        super_res_unet_first_steps_original_checkpoint_to_diffusers_checkpoint
        (super_res_first_model, super_res_checkpoint))
    super_res_last_model = super_res_unet_last_step_model_from_original_config(
        )
    super_res_last_step_checkpoint = (
        super_res_unet_last_step_original_checkpoint_to_diffusers_checkpoint
        (super_res_last_model, super_res_checkpoint))
    del super_res_checkpoint
    load_checkpoint_to_model(super_res_first_steps_checkpoint,
        super_res_first_model, strict=True)
    load_checkpoint_to_model(super_res_last_step_checkpoint,
        super_res_last_model, strict=True)
    print('done loading super resolution unet')
    return super_res_first_model, super_res_last_model
