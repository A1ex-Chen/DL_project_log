def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputs_dir', type=str, required=True,
        help='path to outputs directory')
    parser.add_argument('-d', '--data_dir', type=str, required=True, help=
        'path to data directory')
    parser.add_argument('-e', '--extra_data_dirs', type=str, help=
        'path to extra data directory list')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume_checkpoint', type=str)
    parser.add_argument('--finetune_checkpoint', type=str)
    parser.add_argument('--load_checkpoint', type=str)
    parser.add_argument('--num_workers', type=str)
    parser.add_argument('--visible_devices', type=str)
    parser.add_argument('--needs_freeze_bn', type=str)
    parser.add_argument('--image_resized_width', type=str)
    parser.add_argument('--image_resized_height', type=str)
    parser.add_argument('--image_min_side', type=str)
    parser.add_argument('--image_max_side', type=str)
    parser.add_argument('--image_side_divisor', type=str)
    parser.add_argument('--aug_strategy', type=str, choices=Augmenter.OPTIONS)
    parser.add_argument('--aug_hflip_prob', type=str)
    parser.add_argument('--aug_vflip_prob', type=str)
    parser.add_argument('--aug_rotate90_prob', type=str)
    parser.add_argument('--aug_crop_prob_and_min_max', type=str)
    parser.add_argument('--aug_zoom_prob_and_min_max', type=str)
    parser.add_argument('--aug_scale_prob_and_min_max', type=str)
    parser.add_argument('--aug_translate_prob_and_min_max', type=str)
    parser.add_argument('--aug_rotate_prob_and_min_max', type=str)
    parser.add_argument('--aug_shear_prob_and_min_max', type=str)
    parser.add_argument('--aug_blur_prob_and_min_max', type=str)
    parser.add_argument('--aug_sharpen_prob_and_min_max', type=str)
    parser.add_argument('--aug_color_prob_and_min_max', type=str)
    parser.add_argument('--aug_brightness_prob_and_min_max', type=str)
    parser.add_argument('--aug_grayscale_prob_and_min_max', type=str)
    parser.add_argument('--aug_contrast_prob_and_min_max', type=str)
    parser.add_argument('--aug_noise_prob_and_min_max', type=str)
    parser.add_argument('--aug_resized_crop_prob_and_width_height', type=str)
    parser.add_argument('--batch_size', type=str)
    parser.add_argument('--learning_rate', type=str)
    parser.add_argument('--momentum', type=str)
    parser.add_argument('--weight_decay', type=str)
    parser.add_argument('--clip_grad_base_and_max', type=str)
    parser.add_argument('--step_lr_sizes', type=str)
    parser.add_argument('--step_lr_gamma', type=str)
    parser.add_argument('--warm_up_factor', type=str)
    parser.add_argument('--warm_up_num_iters', type=str)
    parser.add_argument('--num_batches_to_display', type=str)
    parser.add_argument('--num_epochs_to_validate', type=str)
    parser.add_argument('--num_epochs_to_finish', type=str)
    parser.add_argument('--max_num_checkpoints', type=str)
    subparsers = parser.add_subparsers(dest='task', help='task name')
    classification_subparser = subparsers.add_parser(Task.Name.
        CLASSIFICATION.value)
    detection_subparser = subparsers.add_parser(Task.Name.DETECTION.value)
    instance_segmentation_subparser = subparsers.add_parser(Task.Name.
        INSTANCE_SEGMENTATION.value)
    classification_subparser.add_argument('--algorithm', type=str, choices=
        ClassificationAlgorithm.OPTIONS)
    classification_subparser.add_argument('--pretrained', type=str)
    classification_subparser.add_argument('--num_frozen_levels', type=str)
    classification_subparser.add_argument('--eval_center_crop_ratio', type=str)
    detection_subparser.add_argument('--algorithm', type=str, choices=
        DetectionAlgorithm.OPTIONS)
    detection_subparser.add_argument('--backbone', type=str, choices=
        Backbone.OPTIONS)
    detection_subparser.add_argument('--anchor_ratios', type=str)
    detection_subparser.add_argument('--anchor_sizes', type=str)
    detection_subparser.add_argument('--backbone_pretrained', type=str)
    detection_subparser.add_argument('--backbone_num_frozen_levels', type=str)
    detection_subparser.add_argument('--train_rpn_pre_nms_top_n', type=str)
    detection_subparser.add_argument('--train_rpn_post_nms_top_n', type=str)
    detection_subparser.add_argument('--eval_rpn_pre_nms_top_n', type=str)
    detection_subparser.add_argument('--eval_rpn_post_nms_top_n', type=str)
    detection_subparser.add_argument('--num_anchor_samples_per_batch', type=str
        )
    detection_subparser.add_argument('--num_proposal_samples_per_batch',
        type=str)
    detection_subparser.add_argument('--num_detections_per_image', type=str)
    detection_subparser.add_argument('--anchor_smooth_l1_loss_beta', type=str)
    detection_subparser.add_argument('--proposal_smooth_l1_loss_beta', type=str
        )
    detection_subparser.add_argument('--proposal_nms_threshold', type=str)
    detection_subparser.add_argument('--detection_nms_threshold', type=str)
    detection_subparser.add_argument('--eval_quality', type=str)
    instance_segmentation_subparser.add_argument('--algorithm', type=str,
        choices=InstanceSegmentationAlgorithm.OPTIONS)
    checkpoint_id = str(uuid.uuid4())
    args = parser.parse_args()
    path_to_checkpoints_dir = os.path.join(args.outputs_dir, args.task,
        args.tag, args.algorithm if 'algorithm' in args else '', args.
        backbone if 'backbone' in args else '', 'checkpoints-{:s}-{:s}'.
        format(time.strftime('%Y%m%d%H%M%S'), checkpoint_id))
    os.makedirs(path_to_checkpoints_dir)
    task_name = Task.Name(args.task)
    common_args = (task_name.value, path_to_checkpoints_dir, args.data_dir,
        args.extra_data_dirs, args.resume_checkpoint, args.
        finetune_checkpoint, args.load_checkpoint, args.num_workers, args.
        visible_devices, args.needs_freeze_bn, args.image_resized_width,
        args.image_resized_height, args.image_min_side, args.image_max_side,
        args.image_side_divisor, args.aug_strategy, args.aug_hflip_prob,
        args.aug_vflip_prob, args.aug_rotate90_prob, args.
        aug_crop_prob_and_min_max, args.aug_zoom_prob_and_min_max, args.
        aug_scale_prob_and_min_max, args.aug_translate_prob_and_min_max,
        args.aug_rotate_prob_and_min_max, args.aug_shear_prob_and_min_max,
        args.aug_blur_prob_and_min_max, args.aug_sharpen_prob_and_min_max,
        args.aug_color_prob_and_min_max, args.
        aug_brightness_prob_and_min_max, args.
        aug_grayscale_prob_and_min_max, args.aug_contrast_prob_and_min_max,
        args.aug_noise_prob_and_min_max, args.
        aug_resized_crop_prob_and_width_height, args.batch_size, args.
        learning_rate, args.momentum, args.weight_decay, args.
        clip_grad_base_and_max, args.step_lr_sizes, args.step_lr_gamma,
        args.warm_up_factor, args.warm_up_num_iters, args.
        num_batches_to_display, args.num_epochs_to_validate, args.
        num_epochs_to_finish, args.max_num_checkpoints)
    if task_name == Task.Name.CLASSIFICATION:
        config_dict = ClassificationConfig.parse_config_dict(*common_args,
            args.algorithm)
        config = ClassificationConfig(**config_dict)
    elif task_name == Task.Name.DETECTION:
        config_dict = DetectionConfig.parse_config_dict(*common_args, args.
            algorithm, args.backbone, args.anchor_ratios, args.anchor_sizes,
            args.backbone_pretrained, args.backbone_num_frozen_levels, args
            .train_rpn_pre_nms_top_n, args.train_rpn_post_nms_top_n, args.
            eval_rpn_pre_nms_top_n, args.eval_rpn_post_nms_top_n, args.
            num_anchor_samples_per_batch, args.
            num_proposal_samples_per_batch, args.num_detections_per_image,
            args.anchor_smooth_l1_loss_beta, args.
            proposal_smooth_l1_loss_beta, args.proposal_nms_threshold, args
            .detection_nms_threshold, args.eval_quality)
        config = DetectionConfig(**config_dict)
    elif task_name == Task.Name.INSTANCE_SEGMENTATION:
        config_dict = InstanceSegmentationConfig.parse_config_dict(*
            common_args, args.algorithm)
        config = InstanceSegmentationConfig(**config_dict)
    else:
        raise ValueError('Invalid task name')
    terminator = Event()
    _train(config, terminator)
