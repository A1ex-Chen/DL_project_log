def preprocess(self, images: ImageInput, do_resize: Optional[bool]=None,
    size: Optional[Dict[str, int]]=None, resample: PILImageResampling=None,
    do_rescale: Optional[bool]=None, do_center_crop: Optional[bool]=None,
    rescale_factor: Optional[float]=None, do_normalize: Optional[bool]=None,
    image_mean: Optional[Union[float, List[float]]]=None, image_std:
    Optional[Union[float, List[float]]]=None, return_tensors: Optional[
    Union[str, TensorType]]=None, do_convert_rgb: bool=None, data_format:
    ChannelDimension=ChannelDimension.FIRST, input_data_format: Optional[
    Union[str, ChannelDimension]]=None, **kwargs) ->PIL.Image.Image:
    """
        Preprocess an image or batch of images.

        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            size (`Dict[str, int]`, *optional*, defaults to `self.size`):
                Controls the size of the image after `resize`. The shortest edge of the image is resized to
                `size["shortest_edge"]` whilst preserving the aspect ratio. If the longest edge of this resized image
                is > `int(size["shortest_edge"] * (1333 / 800))`, then the image is resized again to make the longest
                edge equal to `int(size["shortest_edge"] * (1333 / 800))`.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image values between [0 - 1].
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to normalize the image by if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to normalize the image by if `do_normalize` is set to `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                    - Unset: Return a list of `np.ndarray`.
                    - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                    - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                    - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                    - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
    do_resize = do_resize if do_resize is not None else self.do_resize
    resample = resample if resample is not None else self.resample
    do_rescale = do_rescale if do_rescale is not None else self.do_rescale
    rescale_factor = (rescale_factor if rescale_factor is not None else
        self.rescale_factor)
    do_normalize = (do_normalize if do_normalize is not None else self.
        do_normalize)
    image_mean = image_mean if image_mean is not None else self.image_mean
    image_std = image_std if image_std is not None else self.image_std
    do_convert_rgb = (do_convert_rgb if do_convert_rgb is not None else
        self.do_convert_rgb)
    do_center_crop = (do_center_crop if do_center_crop is not None else
        self.do_center_crop)
    size = size if size is not None else self.size
    size = get_size_dict(size, default_to_square=False)
    images = make_list_of_images(images)
    if not valid_images(images):
        raise ValueError(
            'Invalid image type. Must be of type PIL.Image.Image, numpy.ndarray, torch.Tensor, tf.Tensor or jax.ndarray.'
            )
    if do_resize and size is None or resample is None:
        raise ValueError(
            'Size and resample must be specified if do_resize is True.')
    if do_rescale and rescale_factor is None:
        raise ValueError(
            'Rescale factor must be specified if do_rescale is True.')
    if do_normalize and (image_mean is None or image_std is None):
        raise ValueError(
            'Image mean and std must be specified if do_normalize is True.')
    if do_convert_rgb:
        images = [convert_to_rgb(image) for image in images]
    images = [to_numpy_array(image) for image in images]
    if is_scaled_image(images[0]) and do_rescale:
        logger.warning_once(
            'It looks like you are trying to rescale already rescaled images. If the input images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again.'
            )
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(images[0])
    if do_resize:
        images = [self.resize(image=image, size=size, resample=resample,
            input_data_format=input_data_format) for image in images]
    if do_rescale:
        images = [self.rescale(image=image, scale=rescale_factor,
            input_data_format=input_data_format) for image in images]
    if do_normalize:
        images = [self.normalize(image=image, mean=image_mean, std=
            image_std, input_data_format=input_data_format) for image in images
            ]
    if do_center_crop:
        images = [self.center_crop(image, size, input_data_format=
            input_data_format) for image in images]
    images = [to_channel_dimension_format(image, data_format,
        input_channel_dim=input_data_format) for image in images]
    encoded_outputs = BatchFeature(data={'pixel_values': images},
        tensor_type=return_tensors)
    return encoded_outputs
