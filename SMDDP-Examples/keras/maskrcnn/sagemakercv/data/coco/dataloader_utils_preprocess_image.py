def preprocess_image(image, boxes, instance_masks, image_size, max_level,
    augment_input_data=False, seed=None):
    image = preprocess_ops.normalize_image(image)
    if augment_input_data:
        image, boxes, instance_masks = augment_image(image=image, boxes=
            boxes, instance_masks=instance_masks, seed=seed)
    image, image_info, boxes, _ = preprocess_ops.resize_and_pad(image=image,
        target_size=image_size, stride=2 ** max_level, boxes=boxes, masks=None)
    return image, image_info, boxes, instance_masks
