def __call__(self, source=None, stream=False, bboxes=None, points=None,
    labels=None, **kwargs):
    """
        Alias for the 'predict' method.

        Args:
            source (str): Path to the image or video file, or a PIL.Image object, or a numpy.ndarray object.
            stream (bool, optional): If True, enables real-time streaming. Defaults to False.
            bboxes (list, optional): List of bounding box coordinates for prompted segmentation. Defaults to None.
            points (list, optional): List of points for prompted segmentation. Defaults to None.
            labels (list, optional): List of labels for prompted segmentation. Defaults to None.

        Returns:
            (list): The model predictions.
        """
    return self.predict(source, stream, bboxes, points, labels, **kwargs)
