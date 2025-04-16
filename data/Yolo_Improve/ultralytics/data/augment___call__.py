def __call__(self, im):
    """
        Transforms an image from a numpy array to a PyTorch tensor, applying optional half-precision and normalization.

        Args:
            im (numpy.ndarray): Input image as a numpy array with shape (H, W, C) in BGR order.

        Returns:
            (torch.Tensor): The transformed image as a PyTorch tensor in float32 or float16, normalized to [0, 1].
        """
    im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])
    im = torch.from_numpy(im)
    im = im.half() if self.half else im.float()
    im /= 255.0
    return im
