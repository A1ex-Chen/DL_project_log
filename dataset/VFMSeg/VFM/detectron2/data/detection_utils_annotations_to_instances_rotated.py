def annotations_to_instances_rotated(annos, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.
    Compared to `annotations_to_instances`, this function is for rotated boxes only

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            Containing fields "gt_boxes", "gt_classes",
            if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [obj['bbox'] for obj in annos]
    target = Instances(image_size)
    boxes = target.gt_boxes = RotatedBoxes(boxes)
    boxes.clip(image_size)
    classes = [obj['category_id'] for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes
    return target
