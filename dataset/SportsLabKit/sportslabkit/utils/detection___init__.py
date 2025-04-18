def __init__(self, camera: Camera, frame_idx: (int | None)=None,
    detection_id: (int | None)=None, detection_confidence: (float | None)=
    None, from_bbox: (BoundingBox | None)=None):
    """Creates a new candidate detection.

        Args:
            camera (Camera): The camera that produced the detection.
            frame_idx (Optional[int], optional): Frame index. Defaults to None.
            detection_id (Optional[int], optional): Dection identification number Defaults to None.
            detection_confidence (Optional[float], optional): Confidence of bounding box. Defaults to None.
            from_bbox (Optional[BoundingBox], optional): `Bounding Box` to instantiate from. Defaults to None.

        Raises:
            NotImplementedError: Support for this instance without bbox is not implemented yet.

        Todo:
            * Implement support for this instance without bbox.
            * Add attribute documentation. Try to inherit from `BoundBox`.
            * Add examples.
        """
    self.camera = camera
    self.frame_idx = frame_idx
    if from_bbox is not None:
        bbox = from_bbox
        super().__init__(image_name=bbox._image_name, class_id=bbox.
            _class_id, coordinates=bbox.get_absolute_bounding_box(),
            img_size=bbox.get_image_size(), bb_type=bbox._bb_type,
            confidence=bbox._confidence)
    else:
        super().__init__()
        self.deteection_id = detection_id
        self.detection_confidence = detection_confidence
        raise NotImplementedError()
    self.px, self.py = camera.video2pitch(np.array([self._x, self._y])
        ).squeeze()
    self.in_range = all([self.camera.x_range[0] <= self.px <= self.camera.
        x_range[-1], self.camera.y_range[0] <= self.py <= self.camera.
        y_range[-1]])
