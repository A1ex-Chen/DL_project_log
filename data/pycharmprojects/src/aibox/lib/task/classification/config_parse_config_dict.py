@staticmethod
def parse_config_dict(task_name: str, path_to_checkpoints_dir: str,
    path_to_data_dir: str, path_to_extra_data_dirs: str=None,
    path_to_resuming_checkpoint: str=None, path_to_finetuning_checkpoint:
    str=None, path_to_loading_checkpoint: str=None, num_workers: str=None,
    visible_devices: str=None, needs_freeze_bn: str=None,
    image_resized_width: str=None, image_resized_height: str=None,
    image_min_side: str=None, image_max_side: str=None, image_side_divisor:
    str=None, aug_strategy: str=None, aug_hflip_prob: str=None,
    aug_vflip_prob: str=None, aug_rotate90_prob: str=None,
    aug_crop_prob_and_min_max: str=None, aug_zoom_prob_and_min_max: str=
    None, aug_scale_prob_and_min_max: str=None,
    aug_translate_prob_and_min_max: str=None, aug_rotate_prob_and_min_max:
    str=None, aug_shear_prob_and_min_max: str=None,
    aug_blur_prob_and_min_max: str=None, aug_sharpen_prob_and_min_max: str=
    None, aug_color_prob_and_min_max: str=None,
    aug_brightness_prob_and_min_max: str=None,
    aug_grayscale_prob_and_min_max: str=None, aug_contrast_prob_and_min_max:
    str=None, aug_noise_prob_and_min_max: str=None,
    aug_resized_crop_prob_and_width_height: str=None, batch_size: str=None,
    learning_rate: str=None, momentum: str=None, weight_decay: str=None,
    clip_grad_base_and_max: str=None, step_lr_sizes: str=None,
    step_lr_gamma: str=None, warm_up_factor: str=None, warm_up_num_iters:
    str=None, num_batches_to_display: str=None, num_epochs_to_validate: str
    =None, num_epochs_to_finish: str=None, max_num_checkpoints: str=None,
    algorithm_name: str=None, pretrained: str=None, num_frozen_levels: str=
    None, eval_center_crop_ratio: str=None) ->Dict[str, Any]:
    config_dict = super(Config, Config).parse_config_dict(task_name,
        path_to_checkpoints_dir, path_to_data_dir, path_to_extra_data_dirs,
        path_to_resuming_checkpoint, path_to_finetuning_checkpoint,
        path_to_loading_checkpoint, num_workers, visible_devices,
        needs_freeze_bn, image_resized_width, image_resized_height,
        image_min_side, image_max_side, image_side_divisor, aug_strategy,
        aug_hflip_prob, aug_vflip_prob, aug_rotate90_prob,
        aug_crop_prob_and_min_max, aug_zoom_prob_and_min_max,
        aug_scale_prob_and_min_max, aug_translate_prob_and_min_max,
        aug_rotate_prob_and_min_max, aug_shear_prob_and_min_max,
        aug_blur_prob_and_min_max, aug_sharpen_prob_and_min_max,
        aug_color_prob_and_min_max, aug_brightness_prob_and_min_max,
        aug_grayscale_prob_and_min_max, aug_contrast_prob_and_min_max,
        aug_noise_prob_and_min_max, aug_resized_crop_prob_and_width_height,
        batch_size, learning_rate, momentum, weight_decay,
        clip_grad_base_and_max, step_lr_sizes, step_lr_gamma,
        warm_up_factor, warm_up_num_iters, num_batches_to_display,
        num_epochs_to_validate, num_epochs_to_finish, max_num_checkpoints)
    assert algorithm_name is not None
    config_dict['algorithm_name'] = Algorithm.Name(algorithm_name)
    if pretrained is not None:
        config_dict['pretrained'] = bool(strtobool(pretrained))
    if num_frozen_levels is not None:
        config_dict['num_frozen_levels'] = int(num_frozen_levels)
    if eval_center_crop_ratio is not None:
        config_dict['eval_center_crop_ratio'] = float(eval_center_crop_ratio)
    return config_dict
