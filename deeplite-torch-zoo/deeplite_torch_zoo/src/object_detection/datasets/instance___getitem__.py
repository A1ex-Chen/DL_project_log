def __getitem__(self, index) ->'Instances':
    """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
    segments = self.segments[index] if len(self.segments) else self.segments
    keypoints = self.keypoints[index] if self.keypoints is not None else None
    bboxes = self.bboxes[index]
    bbox_format = self._bboxes.format
    return Instances(bboxes=bboxes, segments=segments, keypoints=keypoints,
        bbox_format=bbox_format, normalized=self.normalized)
