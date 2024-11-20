def preprocess(self, im):
    """
        Preprocess the input image for model inference.

        The method prepares the input image by applying transformations and normalization.
        It supports both torch.Tensor and list of np.ndarray as input formats.

        Args:
            im (torch.Tensor | List[np.ndarray]): BCHW tensor format or list of HWC numpy arrays.

        Returns:
            (torch.Tensor): The preprocessed image tensor.
        """
    if self.im is not None:
        return self.im
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(self.pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)
    im = im.to(self.device)
    im = im.half() if self.model.fp16 else im.float()
    if not_tensor:
        im = (im - self.mean) / self.std
    return im
