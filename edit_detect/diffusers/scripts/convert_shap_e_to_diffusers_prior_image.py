def prior_image(*, args, checkpoint_map_location):
    print('loading prior_image')
    print(f'load checkpoint from {args.prior_image_checkpoint_path}')
    prior_checkpoint = torch.load(args.prior_image_checkpoint_path,
        map_location=checkpoint_map_location)
    prior_model = prior_image_model_from_original_config()
    prior_diffusers_checkpoint = (
        prior_image_original_checkpoint_to_diffusers_checkpoint(prior_model,
        prior_checkpoint))
    del prior_checkpoint
    load_prior_checkpoint_to_model(prior_diffusers_checkpoint, prior_model)
    print('done loading prior_image')
    return prior_model
