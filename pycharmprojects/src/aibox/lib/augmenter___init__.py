def __init__(self, strategy: Strategy, aug_hflip_prob: float,
    aug_vflip_prob: float, aug_rotate90_prob: float,
    aug_crop_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_zoom_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_scale_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_translate_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_rotate_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_shear_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_blur_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_sharpen_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_color_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_brightness_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_grayscale_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_contrast_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_noise_prob_and_min_max: Tuple[float, Tuple[float, float]],
    aug_resized_crop_prob_and_width_height: Tuple[float, Tuple[int, int]]):
    super().__init__()
    self.strategy = strategy
    self.aug_hflip_prob = aug_hflip_prob
    self.aug_vflip_prob = aug_vflip_prob
    self.aug_rotate90_prob = aug_rotate90_prob
    self.aug_crop_prob_and_min_max = aug_crop_prob_and_min_max
    self.aug_zoom_prob_and_min_max = aug_zoom_prob_and_min_max
    self.aug_scale_prob_and_min_max = aug_scale_prob_and_min_max
    self.aug_translate_prob_and_min_max = aug_translate_prob_and_min_max
    self.aug_rotate_prob_and_min_max = aug_rotate_prob_and_min_max
    self.aug_shear_prob_and_min_max = aug_shear_prob_and_min_max
    self.aug_blur_prob_and_min_max = aug_blur_prob_and_min_max
    self.aug_sharpen_prob_and_min_max = aug_sharpen_prob_and_min_max
    self.aug_color_prob_and_min_max = aug_color_prob_and_min_max
    self.aug_brightness_prob_and_min_max = aug_brightness_prob_and_min_max
    self.aug_grayscale_prob_and_min_max = aug_grayscale_prob_and_min_max
    self.aug_contrast_prob_and_min_max = aug_contrast_prob_and_min_max
    self.aug_noise_prob_and_min_max = aug_noise_prob_and_min_max
    self.aug_resized_crop_prob_and_width_height = (
        aug_resized_crop_prob_and_width_height)
    self.imgaug_transforms = self.build_imgaug_transforms()
    self.albumentations_transforms = self.build_albumentations_transforms()
