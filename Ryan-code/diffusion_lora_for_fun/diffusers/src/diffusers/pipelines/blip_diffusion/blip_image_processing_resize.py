def resize(self, image: np.ndarray, size: Dict[str, int], resample:
    PILImageResampling=PILImageResampling.BICUBIC, data_format: Optional[
    Union[str, ChannelDimension]]=None, input_data_format: Optional[Union[
    str, ChannelDimension]]=None, **kwargs) ->np.ndarray:
    """
        Resize an image to `(size["height"], size["width"])`.

        Args:
            image (`np.ndarray`):
                Image to resize.
            size (`Dict[str, int]`):
                Dictionary in the format `{"height": int, "width": int}` specifying the size of the output image.
            resample (`PILImageResampling`, *optional*, defaults to `PILImageResampling.BICUBIC`):
                `PILImageResampling` filter to use when resizing the image e.g. `PILImageResampling.BICUBIC`.
            data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the output image. If unset, the channel dimension format of the input
                image is used. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        Returns:
            `np.ndarray`: The resized image.
        """
    size = get_size_dict(size)
    if 'height' not in size or 'width' not in size:
        raise ValueError(
            f'The `size` dictionary must contain the keys `height` and `width`. Got {size.keys()}'
            )
    output_size = size['height'], size['width']
    return resize(image, size=output_size, resample=resample, data_format=
        data_format, input_data_format=input_data_format, **kwargs)
