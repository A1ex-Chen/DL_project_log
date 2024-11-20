def build_imgaug_transforms(self) ->List:
    aug_hflip_prob = self.aug_hflip_prob
    aug_vflip_prob = self.aug_vflip_prob
    aug_rotate90_prob = self.aug_rotate90_prob
    aug_crop_prob = self.aug_crop_prob_and_min_max[0]
    aug_zoom_prob = self.aug_zoom_prob_and_min_max[0]
    aug_scale_prob = self.aug_scale_prob_and_min_max[0]
    aug_translate_prob = self.aug_translate_prob_and_min_max[0]
    aug_rotate_prob = self.aug_rotate_prob_and_min_max[0]
    aug_shear_prob = self.aug_shear_prob_and_min_max[0]
    aug_blur_prob = self.aug_blur_prob_and_min_max[0]
    aug_sharpen_prob = self.aug_sharpen_prob_and_min_max[0]
    aug_color_prob = self.aug_color_prob_and_min_max[0]
    aug_brightness_prob = self.aug_brightness_prob_and_min_max[0]
    aug_grayscale_prob = self.aug_grayscale_prob_and_min_max[0]
    aug_contrast_prob = self.aug_contrast_prob_and_min_max[0]
    aug_noise_prob = self.aug_noise_prob_and_min_max[0]
    assert 0 <= aug_hflip_prob <= 1
    assert 0 <= aug_vflip_prob <= 1
    assert 0 <= aug_rotate90_prob <= 1
    assert 0 <= aug_crop_prob <= 1
    assert 0 <= aug_zoom_prob <= 1
    assert 0 <= aug_scale_prob <= 1
    assert 0 <= aug_translate_prob <= 1
    assert 0 <= aug_rotate_prob <= 1
    assert 0 <= aug_shear_prob <= 1
    assert 0 <= aug_blur_prob <= 1
    assert 0 <= aug_sharpen_prob <= 1
    assert 0 <= aug_color_prob <= 1
    assert 0 <= aug_brightness_prob <= 1
    assert 0 <= aug_grayscale_prob <= 1
    assert 0 <= aug_contrast_prob <= 1
    assert 0 <= aug_noise_prob <= 1
    aug_crop_min_max = self.aug_crop_prob_and_min_max[1]
    aug_zoom_min_max = self.aug_zoom_prob_and_min_max[1]
    aug_scale_min_max = self.aug_scale_prob_and_min_max[1]
    aug_translate_min_max = self.aug_translate_prob_and_min_max[1]
    aug_rotate_min_max = self.aug_rotate_prob_and_min_max[1]
    aug_shear_min_max = self.aug_shear_prob_and_min_max[1]
    aug_blur_min_max = self.aug_blur_prob_and_min_max[1]
    aug_sharpen_min_max = self.aug_sharpen_prob_and_min_max[1]
    aug_color_min_max = self.aug_color_prob_and_min_max[1]
    aug_brightness_min_max = self.aug_brightness_prob_and_min_max[1]
    aug_grayscale_min_max = self.aug_grayscale_prob_and_min_max[1]
    aug_contrast_min_max = self.aug_contrast_prob_and_min_max[1]
    aug_noise_min_max = self.aug_noise_prob_and_min_max[1]
    transforms = [Fliplr(p=aug_hflip_prob), Flipud(p=aug_vflip_prob),
        Sometimes(aug_rotate90_prob, Rot90(k=[0, 1, 2, 3], keep_size=False)
        ), Sometimes(aug_crop_prob, Crop(percent=self.denormalize('crop',
        normalized_min_max=aug_crop_min_max))), Sometimes(aug_zoom_prob,
        Affine(scale=self.denormalize('zoom', normalized_min_max=
        aug_zoom_min_max), fit_output=False)), Sometimes(aug_scale_prob,
        Affine(scale=self.denormalize('scale', normalized_min_max=
        aug_scale_min_max), fit_output=True)), Sometimes(aug_translate_prob,
        Affine(translate_percent={'x': self.denormalize('translate',
        normalized_min_max=aug_translate_min_max), 'y': self.denormalize(
        'translate', normalized_min_max=aug_translate_min_max)})),
        Sometimes(aug_rotate_prob, Affine(rotate=self.denormalize('rotate',
        normalized_min_max=aug_rotate_min_max))), Sometimes(aug_shear_prob,
        Affine(shear=self.denormalize('shear', normalized_min_max=
        aug_shear_min_max))), Sometimes(aug_blur_prob, GaussianBlur(sigma=
        self.denormalize('blur', normalized_min_max=aug_blur_min_max))),
        Sometimes(aug_sharpen_prob, Sharpen(alpha=self.denormalize(
        'sharpen', normalized_min_max=aug_sharpen_min_max), lightness=1.0)),
        Sometimes(aug_color_prob, AddToHueAndSaturation(value=self.
        denormalize('color', normalized_min_max=aug_color_min_max))),
        Sometimes(aug_brightness_prob, Add(value=self.denormalize(
        'brightness', normalized_min_max=aug_brightness_min_max))),
        Sometimes(aug_grayscale_prob, Grayscale(alpha=self.denormalize(
        'grayscale', normalized_min_max=aug_grayscale_min_max))), Sometimes
        (aug_contrast_prob, LogContrast(gain=self.denormalize('contrast',
        normalized_min_max=aug_contrast_min_max))), Sometimes(
        aug_noise_prob, SaltAndPepper(p=self.denormalize('noise',
        normalized_min_max=aug_noise_min_max)))]
    return transforms
