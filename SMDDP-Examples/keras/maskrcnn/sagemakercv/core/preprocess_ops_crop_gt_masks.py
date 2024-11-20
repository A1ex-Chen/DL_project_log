def crop_gt_masks(instance_masks, boxes, gt_mask_size, image_size):
    """Crops the ground truth binary masks and resize to fixed-size masks."""
    num_masks = tf.shape(input=instance_masks)[0]
    scale_sizes = tf.convert_to_tensor(value=[image_size[0], image_size[1]] *
        2, dtype=tf.float32)
    boxes = boxes / scale_sizes
    cropped_gt_masks = tf.image.crop_and_resize(image=instance_masks, boxes
        =boxes, box_indices=tf.range(num_masks, dtype=tf.int32), crop_size=
        [gt_mask_size, gt_mask_size], method='bilinear')[:, :, :, 0]
    cropped_gt_masks = tf.pad(tensor=cropped_gt_masks, paddings=tf.constant
        ([[0, 0], [2, 2], [2, 2]]), mode='CONSTANT', constant_values=0.0)
    return cropped_gt_masks
