@classmethod
def concatenate(cls, instances_list: List['Instances'], axis=0) ->'Instances':
    """
        Concatenates a list of Instances objects into a single Instances object.

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated. Defaults to 0.

        Returns:
            Instances: A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
        """
    assert isinstance(instances_list, (list, tuple))
    if not instances_list:
        return cls(np.empty(0))
    assert all(isinstance(instance, Instances) for instance in instances_list)
    if len(instances_list) == 1:
        return instances_list[0]
    use_keypoint = instances_list[0].keypoints is not None
    bbox_format = instances_list[0]._bboxes.format
    normalized = instances_list[0].normalized
    cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis
        =axis)
    cat_segments = np.concatenate([b.segments for b in instances_list],
        axis=axis)
    cat_keypoints = np.concatenate([b.keypoints for b in instances_list],
        axis=axis) if use_keypoint else None
    return cls(cat_boxes, cat_segments, cat_keypoints, bbox_format, normalized)
