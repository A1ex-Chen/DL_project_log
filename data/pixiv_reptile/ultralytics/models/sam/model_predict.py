def predict(self, source, stream=False, bboxes=None, points=None, labels=
    None, **kwargs):
    """
        Performs segmentation prediction on the given image or video source.

        Args:
            source (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
            stream (bool, optional): If True, enables real-time streaming. Defaults to False.
            bboxes (list, optional): List of bounding box coordinates for prompted segmentation. Defaults to None.
            points (list, optional): List of points for prompted segmentation. Defaults to None.
            labels (list, optional): List of labels for prompted segmentation. Defaults to None.

        Returns:
            (list): The model predictions.
        """
    overrides = dict(conf=0.25, task='segment', mode='predict', imgsz=1024)
    kwargs.update(overrides)
    prompts = dict(bboxes=bboxes, points=points, labels=labels)
    return super().predict(source, stream, prompts=prompts, **kwargs)
