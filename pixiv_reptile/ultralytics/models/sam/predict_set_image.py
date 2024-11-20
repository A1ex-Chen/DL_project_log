def set_image(self, image):
    """
        Preprocesses and sets a single image for inference.

        This function sets up the model if not already initialized, configures the data source to the specified image,
        and preprocesses the image for feature extraction. Only one image can be set at a time.

        Args:
            image (str | np.ndarray): Image file path as a string, or a np.ndarray image read by cv2.

        Raises:
            AssertionError: If more than one image is set.
        """
    if self.model is None:
        model = build_sam(self.args.model)
        self.setup_model(model)
    self.setup_source(image)
    assert len(self.dataset
        ) == 1, '`set_image` only supports setting one image!'
    for batch in self.dataset:
        im = self.preprocess(batch[1])
        self.features = self.model.image_encoder(im)
        self.im = im
        break
