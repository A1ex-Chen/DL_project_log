def augment_image(image, boxes, instance_masks, seed):
    flipped_results = preprocess_ops.random_horizontal_flip(image, boxes=
        boxes, masks=instance_masks, seed=seed)
    if instance_masks is not None:
        image, boxes, instance_masks = flipped_results
    else:
        image, boxes = flipped_results
    image = preprocess_ops.add_noise(image, std=0.2, seed=seed)
    return image, boxes, instance_masks
