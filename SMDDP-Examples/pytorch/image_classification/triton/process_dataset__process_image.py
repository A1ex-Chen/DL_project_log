def _process_image(image_file, target_size):
    image = Image.open(image_file)
    original_size = image.size
    scale_factor = max(_RESIZE_MIN / original_size[0], _RESIZE_MIN /
        original_size[1])
    resize_to = int(original_size[0] * scale_factor), int(original_size[1] *
        scale_factor)
    resized_image = image.resize(resize_to)
    left, upper = (resize_to[0] - target_size[0]) // 2, (resize_to[1] -
        target_size[1]) // 2
    cropped_image = resized_image.crop((left, upper, left + target_size[0],
        upper + target_size[1]))
    return cropped_image
