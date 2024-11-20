def annotations_to_instances(annos, image_size, mask_format='polygon'):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = np.stack([BoxMode.convert(obj['bbox'], obj['bbox_mode'],
        BoxMode.XYXY_ABS) for obj in annos]) if len(annos) else np.zeros((0, 4)
        )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    classes = [int(obj['category_id']) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    if len(annos) and 'segmentation' in annos[0]:
        segms = [obj['segmentation'] for obj in annos]
        if mask_format == 'polygon':
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                    ) from e
        else:
            assert mask_format == 'bitmask', mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, 'Expect segmentation of 2 dimensions, got {}.'.format(
                        segm.ndim)
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!Supported types are: polygons as list[list[float] or ndarray], COCO-style RLE as a dict, or a binary segmentation mask  in a 2D numpy array of shape HxW."
                        .format(type(segm)))
            masks = BitMasks(torch.stack([torch.from_numpy(np.
                ascontiguousarray(x).copy()) for x in masks]))
        target.gt_masks = masks
    if len(annos) and 'keypoints' in annos[0]:
        kpts = [obj.get('keypoints', []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)
    return target
