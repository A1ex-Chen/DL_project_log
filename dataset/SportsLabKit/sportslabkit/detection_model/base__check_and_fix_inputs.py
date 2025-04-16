def _check_and_fix_inputs(self, inputs):
    """Check input type and shape.

        Acceptable input types are numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays.
        """
    if isinstance(inputs, (list, tuple, np.ndarray, torch.Tensor)):
        self.input_is_batched = isinstance(inputs, (list, tuple)) or hasattr(
            inputs, 'ndim') and inputs.ndim == 4
        if not self.input_is_batched:
            inputs = [inputs]
    else:
        inputs = [inputs]
    imgs = []
    for img in inputs:
        img = self.read_image(img)
        imgs.append(img)
    return imgs
