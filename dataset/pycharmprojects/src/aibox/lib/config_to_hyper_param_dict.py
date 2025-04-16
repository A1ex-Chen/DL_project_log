def to_hyper_param_dict(self) ->Dict[str, Union[int, float, str, bool, Tensor]
    ]:
    hyper_param_dict = {'num_workers': self.num_workers, 'visible_devices':
        str(self.visible_devices), 'needs_freeze_bn': self.needs_freeze_bn,
        'image_resized_width': self.image_resized_width,
        'image_resized_height': self.image_resized_height, 'image_min_side':
        self.image_min_side, 'image_max_side': self.image_max_side,
        'image_side_divisor': self.image_side_divisor, 'aug_strategy': str(
        self.aug_strategy), 'aug_hflip_prob': self.aug_hflip_prob,
        'aug_vflip_prob': self.aug_vflip_prob, 'aug_rotate90_prob': self.
        aug_rotate90_prob, 'aug_crop_prob_and_min_max': str(self.
        aug_crop_prob_and_min_max), 'aug_zoom_prob_and_min_max': str(self.
        aug_zoom_prob_and_min_max), 'aug_scale_prob_and_min_max': str(self.
        aug_scale_prob_and_min_max), 'aug_translate_prob_and_min_max': str(
        self.aug_translate_prob_and_min_max), 'aug_rotate_prob_and_min_max':
        str(self.aug_rotate_prob_and_min_max), 'aug_shear_prob_and_min_max':
        str(self.aug_shear_prob_and_min_max), 'aug_blur_prob_and_min_max':
        str(self.aug_blur_prob_and_min_max), 'aug_sharpen_prob_and_min_max':
        str(self.aug_sharpen_prob_and_min_max),
        'aug_color_prob_and_min_max': str(self.aug_color_prob_and_min_max),
        'aug_brightness_prob_and_min_max': str(self.
        aug_brightness_prob_and_min_max), 'aug_grayscale_prob_and_min_max':
        str(self.aug_grayscale_prob_and_min_max),
        'aug_contrast_prob_and_min_max': str(self.
        aug_contrast_prob_and_min_max), 'aug_noise_prob_and_min_max': str(
        self.aug_noise_prob_and_min_max),
        'aug_resized_crop_prob_and_width_height': str(self.
        aug_resized_crop_prob_and_width_height), 'batch_size': self.
        batch_size, 'learning_rate': self.learning_rate, 'momentum': self.
        momentum, 'weight_decay': self.weight_decay,
        'clip_grad_base_and_max': str(self.clip_grad_base_and_max),
        'step_lr_sizes': str(self.step_lr_sizes), 'step_lr_gamma': self.
        step_lr_gamma, 'warm_up_factor': self.warm_up_factor,
        'warm_up_num_iters': self.warm_up_num_iters, 'num_epochs_to_finish':
        self.num_epochs_to_finish}
    return hyper_param_dict
