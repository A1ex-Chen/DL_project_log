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
    algorithm_name: str=None, backbone_name: str=None, anchor_ratios: str=
    None, anchor_sizes: str=None, backbone_pretrained: str=None,
    backbone_num_frozen_levels: str=None, train_rpn_pre_nms_top_n: str=None,
    train_rpn_post_nms_top_n: str=None, eval_rpn_pre_nms_top_n: str=None,
    eval_rpn_post_nms_top_n: str=None, num_anchor_samples_per_batch: str=
    None, num_proposal_samples_per_batch: str=None,
    num_detections_per_image: str=None, anchor_smooth_l1_loss_beta: str=
    None, proposal_smooth_l1_loss_beta: str=None, proposal_nms_threshold:
    str=None, detection_nms_threshold: str=None, eval_quality: str=None
    ) ->Dict[str, Any]:
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
    assert backbone_name is not None
    algorithm_name = Algorithm.Name(algorithm_name)
    backbone_name = Backbone.Name(backbone_name)
    if algorithm_name == Algorithm.Name.FASTER_RCNN:
        default_anchor_ratios = [(1, 2), (1, 1), (2, 1)]
        default_anchor_sizes = [128, 256, 512]
        default_train_rpn_pre_nms_top_n = 12000
        default_train_rpn_post_nms_top_n = 2000
        default_eval_rpn_pre_nms_top_n = 6000
        default_eval_rpn_post_nms_top_n = 1000
    elif algorithm_name == Algorithm.Name.FPN:
        default_anchor_ratios = [(1, 2), (1, 1), (2, 1)]
        default_anchor_sizes = [128]
        default_train_rpn_pre_nms_top_n = 2000
        default_train_rpn_post_nms_top_n = 2000
        default_eval_rpn_pre_nms_top_n = 1000
        default_eval_rpn_post_nms_top_n = 1000
    elif algorithm_name == Algorithm.Name.TORCH_FPN:
        default_anchor_ratios = [(1, 2), (1, 1), (2, 1)]
        default_anchor_sizes = [32, 64, 128, 256, 512]
        default_train_rpn_pre_nms_top_n = 2000
        default_train_rpn_post_nms_top_n = 2000
        default_eval_rpn_pre_nms_top_n = 1000
        default_eval_rpn_post_nms_top_n = 1000
    else:
        raise ValueError
    config_dict['algorithm_name'] = Algorithm.Name(algorithm_name)
    config_dict['backbone_name'] = Backbone.Name(backbone_name)
    config_dict['anchor_ratios'] = literal_eval(anchor_ratios
        ) if anchor_ratios is not None else default_anchor_ratios
    config_dict['anchor_sizes'] = literal_eval(anchor_sizes
        ) if anchor_sizes is not None else default_anchor_sizes
    if backbone_pretrained is not None:
        config_dict['backbone_pretrained'] = bool(strtobool(
            backbone_pretrained))
    if backbone_num_frozen_levels is not None:
        config_dict['backbone_num_frozen_levels'] = int(
            backbone_num_frozen_levels)
    config_dict['train_rpn_pre_nms_top_n'] = (int(train_rpn_pre_nms_top_n) if
        train_rpn_pre_nms_top_n is not None else
        default_train_rpn_pre_nms_top_n)
    config_dict['train_rpn_post_nms_top_n'] = (int(train_rpn_post_nms_top_n
        ) if train_rpn_post_nms_top_n is not None else
        default_train_rpn_post_nms_top_n)
    config_dict['eval_rpn_pre_nms_top_n'] = int(eval_rpn_pre_nms_top_n
        ) if eval_rpn_pre_nms_top_n is not None else default_eval_rpn_pre_nms_top_n
    config_dict['eval_rpn_post_nms_top_n'] = (int(eval_rpn_post_nms_top_n) if
        eval_rpn_post_nms_top_n is not None else
        default_eval_rpn_post_nms_top_n)
    if num_anchor_samples_per_batch is not None:
        config_dict['num_anchor_samples_per_batch'] = int(
            num_anchor_samples_per_batch)
    if num_proposal_samples_per_batch is not None:
        config_dict['num_proposal_samples_per_batch'] = int(
            num_proposal_samples_per_batch)
    if num_detections_per_image is not None:
        config_dict['num_detections_per_image'] = int(num_detections_per_image)
    if anchor_smooth_l1_loss_beta is not None:
        config_dict['anchor_smooth_l1_loss_beta'] = float(
            anchor_smooth_l1_loss_beta)
    if proposal_smooth_l1_loss_beta is not None:
        config_dict['proposal_smooth_l1_loss_beta'] = float(
            proposal_smooth_l1_loss_beta)
    if proposal_nms_threshold is not None:
        config_dict['proposal_nms_threshold'] = float(proposal_nms_threshold)
    if detection_nms_threshold is not None:
        config_dict['detection_nms_threshold'] = float(detection_nms_threshold)
    if eval_quality is not None:
        config_dict['eval_quality'] = Evaluator.Evaluation.Quality(eval_quality
            )
    return config_dict
