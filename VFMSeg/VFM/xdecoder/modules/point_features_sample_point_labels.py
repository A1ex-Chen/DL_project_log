def sample_point_labels(instances, point_coords):
    """
    Sample point labels from ground truth mask given point_coords.

    Args:
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. So, i_th elememt of the list contains R_i objects and R_1 + ... + R_N is
            equal to R. The ground-truth gt_masks in each instance will be used to compute labels.
        points_coords (Tensor): A tensor of shape (R, P, 2), where R is the total number of
            instances and P is the number of points for each instance. The coordinates are in
            the absolute image pixel coordinate space, i.e. [0, H] x [0, W].

    Returns:
        Tensor: A tensor of shape (R, P) that contains the labels of P sampled points.
    """
    with torch.no_grad():
        gt_mask_logits = []
        point_coords_splits = torch.split(point_coords, [len(
            instances_per_image) for instances_per_image in instances])
        for i, instances_per_image in enumerate(instances):
            if len(instances_per_image) == 0:
                continue
            assert isinstance(instances_per_image.gt_masks, BitMasks
                ), "Point head works with GT in 'bitmask' format. Set INPUT.MASK_FORMAT to 'bitmask'."
            gt_bit_masks = instances_per_image.gt_masks.tensor
            h, w = instances_per_image.gt_masks.image_size
            scale = torch.tensor([w, h], dtype=torch.float, device=
                gt_bit_masks.device)
            points_coord_grid_sample_format = point_coords_splits[i] / scale
            gt_mask_logits.append(point_sample(gt_bit_masks.to(torch.
                float32).unsqueeze(1), points_coord_grid_sample_format,
                align_corners=False).squeeze(1))
    point_labels = cat(gt_mask_logits)
    return point_labels
