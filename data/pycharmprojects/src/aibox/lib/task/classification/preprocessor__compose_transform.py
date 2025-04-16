def _compose_transform(self, is_train_or_eval: bool, resized_width: int,
    resized_height: int, right_pad: int, bottom_pad: int) ->transforms.Compose:
    if is_train_or_eval or self.eval_center_crop_ratio == 1:
        return super()._compose_transform(is_train_or_eval, resized_width,
            resized_height, right_pad, bottom_pad)
    else:
        center_crop_width = int(resized_width * self.eval_center_crop_ratio)
        center_crop_height = int(resized_height * self.eval_center_crop_ratio)
        transform = transforms.Compose([transforms.Resize(size=(
            resized_height, resized_width)), transforms.CenterCrop(size=(
            center_crop_height, center_crop_width)), transforms.Pad(padding
            =(0, 0, right_pad, bottom_pad), fill=0)])
        return transform
