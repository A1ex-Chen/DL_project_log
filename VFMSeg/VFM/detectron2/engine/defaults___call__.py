def __call__(self, original_image):
    """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
    with torch.no_grad():
        if self.input_format == 'RGB':
            original_image = original_image[:, :, ::-1]
        height, width = original_image.shape[:2]
        image = self.aug.get_transform(original_image).apply_image(
            original_image)
        image = torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
        inputs = {'image': image, 'height': height, 'width': width}
        predictions = self.model([inputs])[0]
        return predictions
