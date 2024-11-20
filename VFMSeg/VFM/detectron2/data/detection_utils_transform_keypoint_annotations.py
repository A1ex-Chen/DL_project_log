def transform_keypoint_annotations(keypoints, transforms, image_size,
    keypoint_hflip_indices=None):
    """
    Transform keypoint annotations of an image.
    If a keypoint is transformed out of image boundary, it will be marked "unlabeled" (visibility=0)

    Args:
        keypoints (list[float]): Nx3 float in Detectron2's Dataset format.
            Each point is represented by (x, y, visibility).
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
            When `transforms` includes horizontal flip, will use the index
            mapping to flip keypoints.
    """
    keypoints = np.asarray(keypoints, dtype='float64').reshape(-1, 3)
    keypoints_xy = transforms.apply_coords(keypoints[:, :2])
    inside = (keypoints_xy >= np.array([0, 0])) & (keypoints_xy <= np.array
        (image_size[::-1]))
    inside = inside.all(axis=1)
    keypoints[:, :2] = keypoints_xy
    keypoints[:, 2][~inside] = 0
    do_hflip = sum(isinstance(t, T.HFlipTransform) for t in transforms.
        transforms) % 2 == 1
    if do_hflip:
        if keypoint_hflip_indices is None:
            raise ValueError(
                'Cannot flip keypoints without providing flip indices!')
        if len(keypoints) != len(keypoint_hflip_indices):
            raise ValueError(
                'Keypoint data has {} points, but metadata contains {} points!'
                .format(len(keypoints), len(keypoint_hflip_indices)))
        keypoints = keypoints[np.asarray(keypoint_hflip_indices, dtype=np.
            int32), :]
    keypoints[keypoints[:, 2] == 0] = 0
    return keypoints
