def transform_instance_annotations(annotation, transforms, image_size, *,
    keypoint_hflip_indices=None):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    bbox = BoxMode.convert(annotation['bbox'], annotation['bbox_mode'],
        BoxMode.XYXY_ABS)
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation['bbox'] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation['bbox_mode'] = BoxMode.XYXY_ABS
    if 'segmentation' in annotation:
        segm = annotation['segmentation']
        if isinstance(segm, list):
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation['segmentation'] = [p.reshape(-1) for p in transforms
                .apply_polygons(polygons)]
        elif isinstance(segm, dict):
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation['segmentation'] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!Supported types are: polygons as list[list[float] or ndarray], COCO-style RLE as a dict."
                .format(type(segm)))
    if 'keypoints' in annotation:
        keypoints = transform_keypoint_annotations(annotation['keypoints'],
            transforms, image_size, keypoint_hflip_indices)
        annotation['keypoints'] = keypoints
    return annotation
