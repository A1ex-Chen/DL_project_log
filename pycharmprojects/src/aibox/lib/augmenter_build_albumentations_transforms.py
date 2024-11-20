def build_albumentations_transforms(self) ->List:
    aug_resized_crop_prob = self.aug_resized_crop_prob_and_width_height[0]
    assert 0 <= aug_resized_crop_prob <= 1
    aug_resized_crop_width_height = (self.
        aug_resized_crop_prob_and_width_height[1])
    transforms = [A.RandomResizedCrop(height=aug_resized_crop_width_height[
        1], width=aug_resized_crop_width_height[0], p=aug_resized_crop_prob)]
    return transforms
