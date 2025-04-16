def process_gt_masks_for_training(instance_masks, boxes, gt_mask_size,
    padded_image_size, max_num_instances):
    cropped_gt_masks = preprocess_ops.crop_gt_masks(instance_masks=
        instance_masks, boxes=boxes, gt_mask_size=gt_mask_size, image_size=
        padded_image_size)
    cropped_gt_masks = preprocess_ops.pad_to_fixed_size(data=
        cropped_gt_masks, pad_value=-1, output_shape=[max_num_instances, (
        gt_mask_size + 4) ** 2])
    return tf.reshape(cropped_gt_masks, [max_num_instances, gt_mask_size + 
        4, gt_mask_size + 4])
