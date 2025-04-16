def preprocess(self, rgb: Union[torch.Tensor, PIL.Image.Image, np.ndarray],
    depth: Union[torch.Tensor, PIL.Image.Image, np.ndarray], height:
    Optional[int]=None, width: Optional[int]=None, target_res: Optional[int
    ]=None) ->torch.Tensor:
    """
        Preprocess the image input. Accepted formats are PIL images, NumPy arrays or PyTorch tensors.
        """
    supported_formats = PIL.Image.Image, np.ndarray, torch.Tensor
    if self.config.do_convert_grayscale and isinstance(rgb, (torch.Tensor,
        np.ndarray)) and rgb.ndim == 3:
        raise Exception('This is not yet supported')
    if isinstance(rgb, supported_formats):
        rgb = [rgb]
        depth = [depth]
    elif not (isinstance(rgb, list) and all(isinstance(i, supported_formats
        ) for i in rgb)):
        raise ValueError(
            f"Input is in incorrect format: {[type(i) for i in rgb]}. Currently, we only support {', '.join(supported_formats)}"
            )
    if isinstance(rgb[0], PIL.Image.Image):
        if self.config.do_convert_rgb:
            raise Exception('This is not yet supported')
        if self.config.do_resize or target_res:
            height, width = self.get_default_height_width(rgb[0], height, width
                ) if not target_res else target_res
            rgb = [self.resize(i, height, width) for i in rgb]
            depth = [self.resize(i, height, width) for i in depth]
        rgb = self.pil_to_numpy(rgb)
        rgb = self.numpy_to_pt(rgb)
        depth = self.depth_pil_to_numpy(depth)
        depth = self.numpy_to_pt(depth)
    elif isinstance(rgb[0], np.ndarray):
        rgb = np.concatenate(rgb, axis=0) if rgb[0].ndim == 4 else np.stack(rgb
            , axis=0)
        rgb = self.numpy_to_pt(rgb)
        height, width = self.get_default_height_width(rgb, height, width)
        if self.config.do_resize:
            rgb = self.resize(rgb, height, width)
        depth = np.concatenate(depth, axis=0) if rgb[0
            ].ndim == 4 else np.stack(depth, axis=0)
        depth = self.numpy_to_pt(depth)
        height, width = self.get_default_height_width(depth, height, width)
        if self.config.do_resize:
            depth = self.resize(depth, height, width)
    elif isinstance(rgb[0], torch.Tensor):
        raise Exception('This is not yet supported')
    do_normalize = self.config.do_normalize
    if rgb.min() < 0 and do_normalize:
        warnings.warn(
            f'Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] when passing as pytorch tensor or numpy Array. You passed `image` with value range [{rgb.min()},{rgb.max()}]'
            , FutureWarning)
        do_normalize = False
    if do_normalize:
        rgb = self.normalize(rgb)
        depth = self.normalize(depth)
    if self.config.do_binarize:
        rgb = self.binarize(rgb)
        depth = self.binarize(depth)
    return rgb, depth
