def preprocess(self, im):
    """
        Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
    not_tensor = not isinstance(im, torch.Tensor)
    if not_tensor:
        im = np.stack(self.pre_transform(im))
        im = im[..., ::-1].transpose((0, 3, 1, 2))
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im)
    im = im.to(self.device)
    im = im.half() if self.model.fp16 else im.float()
    if not_tensor:
        im /= 255
    return im
