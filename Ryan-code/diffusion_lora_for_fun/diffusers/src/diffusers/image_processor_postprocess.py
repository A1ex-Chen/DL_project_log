def postprocess(self, image: torch.Tensor, output_type: str='pil',
    do_denormalize: Optional[List[bool]]=None) ->Union[PIL.Image.Image, np.
    ndarray, torch.Tensor]:
    """
        Postprocess the image output from tensor to `output_type`.

        Args:
            image (`torch.Tensor`):
                The image input, should be a pytorch tensor with shape `B x C x H x W`.
            output_type (`str`, *optional*, defaults to `pil`):
                The output type of the image, can be one of `pil`, `np`, `pt`, `latent`.
            do_denormalize (`List[bool]`, *optional*, defaults to `None`):
                Whether to denormalize the image to [0,1]. If `None`, will use the value of `do_normalize` in the
                `VaeImageProcessor` config.

        Returns:
            `PIL.Image.Image`, `np.ndarray` or `torch.Tensor`:
                The postprocessed image.
        """
    if not isinstance(image, torch.Tensor):
        raise ValueError(
            f'Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor'
            )
    if output_type not in ['latent', 'pt', 'np', 'pil']:
        deprecation_message = (
            f'the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: `pil`, `np`, `pt`, `latent`'
            )
        deprecate('Unsupported output_type', '1.0.0', deprecation_message,
            standard_warn=False)
        output_type = 'np'
    if do_denormalize is None:
        do_denormalize = [self.config.do_normalize] * image.shape[0]
    image = torch.stack([(self.denormalize(image[i]) if do_denormalize[i] else
        image[i]) for i in range(image.shape[0])])
    image = self.pt_to_numpy(image)
    if output_type == 'np':
        if image.shape[-1] == 6:
            image_depth = np.stack([self.rgblike_to_depthmap(im[:, :, 3:]) for
                im in image], axis=0)
        else:
            image_depth = image[:, :, :, 3:]
        return image[:, :, :, :3], image_depth
    if output_type == 'pil':
        return self.numpy_to_pil(image), self.numpy_to_depth(image)
    else:
        raise Exception(f'This type {output_type} is not supported')
