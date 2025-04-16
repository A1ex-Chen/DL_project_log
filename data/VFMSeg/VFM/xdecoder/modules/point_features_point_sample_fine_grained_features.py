def point_sample_fine_grained_features(features_list, feature_scales, boxes,
    point_coords):
    """
    Get features from feature maps in `features_list` that correspond to specific point coordinates
        inside each bounding box from `boxes`.

    Args:
        features_list (list[Tensor]): A list of feature map tensors to get features from.
        feature_scales (list[float]): A list of scales for tensors in `features_list`.
        boxes (list[Boxes]): A list of I Boxes  objects that contain R_1 + ... + R_I = R boxes all
            together.
        point_coords (Tensor): A tensor of shape (R, P, 2) that contains
            [0, 1] x [0, 1] box-normalized coordinates of the P sampled points.

    Returns:
        point_features (Tensor): A tensor of shape (R, C, P) that contains features sampled
            from all features maps in feature_list for P sampled points for all R boxes in `boxes`.
        point_coords_wrt_image (Tensor): A tensor of shape (R, P, 2) that contains image-level
            coordinates of P points.
    """
    cat_boxes = Boxes.cat(boxes)
    num_boxes = [b.tensor.size(0) for b in boxes]
    point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes.tensor,
        point_coords)
    split_point_coords_wrt_image = torch.split(point_coords_wrt_image,
        num_boxes)
    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(
        split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = shapes_to_tensor([w, h]) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale.to(
                feature_map.device)
            point_features_per_image.append(point_sample(feature_map[
                idx_img].unsqueeze(0), point_coords_scaled.unsqueeze(0),
                align_corners=False).squeeze(0).transpose(1, 0))
        point_features.append(cat(point_features_per_image, dim=1))
    return cat(point_features, dim=0), point_coords_wrt_image
