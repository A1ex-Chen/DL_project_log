def pre_transform(self, im):
    """
        Perform initial transformations on the input image for preprocessing.

        The method applies transformations such as resizing to prepare the image for further preprocessing.
        Currently, batched inference is not supported; hence the list length should be 1.

        Args:
            im (List[np.ndarray]): List containing images in HWC numpy array format.

        Returns:
            (List[np.ndarray]): List of transformed images.
        """
    assert len(im
        ) == 1, 'SAM model does not currently support batched inference'
    letterbox = LetterBox(self.args.imgsz, auto=False, center=False)
    return [letterbox(image=x) for x in im]
