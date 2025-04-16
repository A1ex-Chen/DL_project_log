def prior(*, args, checkpoint_map_location):
    print('loading prior')
    prior_checkpoint = torch.load(args.prior_checkpoint_path, map_location=
        checkpoint_map_location)
    clip_stats_checkpoint = torch.load(args.clip_stat_path, map_location=
        checkpoint_map_location)
    prior_model = prior_model_from_original_config()
    prior_diffusers_checkpoint = (
        prior_original_checkpoint_to_diffusers_checkpoint(prior_model,
        prior_checkpoint, clip_stats_checkpoint))
    del prior_checkpoint
    del clip_stats_checkpoint
    load_checkpoint_to_model(prior_diffusers_checkpoint, prior_model,
        strict=True)
    print('done loading prior')
    return prior_model
