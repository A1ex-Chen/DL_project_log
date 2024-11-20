def stable_unclip_image_noising_components(original_config, clip_stats_path:
    Optional[str]=None, device: Optional[str]=None):
    """
    Returns the noising components for the img2img and txt2img unclip pipelines.

    Converts the stability noise augmentor into
    1. a `StableUnCLIPImageNormalizer` for holding the CLIP stats
    2. a `DDPMScheduler` for holding the noise schedule

    If the noise augmentor config specifies a clip stats path, the `clip_stats_path` must be provided.
    """
    noise_aug_config = original_config['model']['params']['noise_aug_config']
    noise_aug_class = noise_aug_config['target']
    noise_aug_class = noise_aug_class.split('.')[-1]
    if noise_aug_class == 'CLIPEmbeddingNoiseAugmentation':
        noise_aug_config = noise_aug_config.params
        embedding_dim = noise_aug_config.timestep_dim
        max_noise_level = noise_aug_config.noise_schedule_config.timesteps
        beta_schedule = noise_aug_config.noise_schedule_config.beta_schedule
        image_normalizer = StableUnCLIPImageNormalizer(embedding_dim=
            embedding_dim)
        image_noising_scheduler = DDPMScheduler(num_train_timesteps=
            max_noise_level, beta_schedule=beta_schedule)
        if 'clip_stats_path' in noise_aug_config:
            if clip_stats_path is None:
                raise ValueError(
                    'This stable unclip config requires a `clip_stats_path`')
            clip_mean, clip_std = torch.load(clip_stats_path, map_location=
                device)
            clip_mean = clip_mean[None, :]
            clip_std = clip_std[None, :]
            clip_stats_state_dict = {'mean': clip_mean, 'std': clip_std}
            image_normalizer.load_state_dict(clip_stats_state_dict)
    else:
        raise NotImplementedError(
            f'Unknown noise augmentor class: {noise_aug_class}')
    return image_normalizer, image_noising_scheduler
