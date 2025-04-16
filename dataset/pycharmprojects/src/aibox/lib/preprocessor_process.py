def process(self, image: Union[PIL.Image.Image, Tensor], is_train_or_eval: bool
    ) ->Tuple[Tensor, Dict[str, Any]]:
    if isinstance(image, PIL.Image.Image):
        image = to_tensor(image)
    image_width, image_height = image.shape[2], image.shape[1]
    scale_for_width = (1 if self.image_resized_width == -1 else self.
        image_resized_width / image_width)
    scale_for_height = (1 if self.image_resized_height == -1 else self.
        image_resized_height / image_height)
    if self.image_min_side == -1:
        scale_for_shorter_side = 1
    else:
        scale_for_shorter_side = self.image_min_side / min(image_width *
            scale_for_width, image_height * scale_for_height)
    if self.image_max_side == -1:
        scale_for_longer_side = 1
    else:
        longer_side_after_scaling = max(image_width * scale_for_width, 
            image_height * scale_for_height) * scale_for_shorter_side
        scale_for_longer_side = (self.image_max_side /
            longer_side_after_scaling if longer_side_after_scaling > self.
            image_max_side else 1)
    scale_for_width *= scale_for_shorter_side * scale_for_longer_side
    scale_for_height *= scale_for_shorter_side * scale_for_longer_side
    scaled_image_width = round(image_width * scale_for_width)
    scaled_image_height = round(image_height * scale_for_height)
    image_right_pad = int(ceil(scaled_image_width / self.image_side_divisor
        ) * self.image_side_divisor) - scaled_image_width
    image_bottom_pad = int(ceil(scaled_image_height / self.
        image_side_divisor) * self.image_side_divisor) - scaled_image_height
    transform = self._compose_transform(is_train_or_eval, resized_width=
        scaled_image_width, resized_height=scaled_image_height, right_pad=
        image_right_pad, bottom_pad=image_bottom_pad)
    processed_image = transform(image)
    process_dict = {self.PROCESS_KEY_IS_TRAIN_OR_EVAL: is_train_or_eval,
        self.PROCESS_KEY_ORIGIN_WIDTH: image_width, self.
        PROCESS_KEY_ORIGIN_HEIGHT: image_height, self.
        PROCESS_KEY_WIDTH_SCALE: scale_for_width, self.
        PROCESS_KEY_HEIGHT_SCALE: scale_for_height, self.
        PROCESS_KEY_RIGHT_PAD: image_right_pad, self.PROCESS_KEY_BOTTOM_PAD:
        image_bottom_pad}
    return processed_image, process_dict
