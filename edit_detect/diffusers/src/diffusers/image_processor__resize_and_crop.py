def _resize_and_crop(self, image: PIL.Image.Image, width: int, height: int
    ) ->PIL.Image.Image:
    """
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, cropping the excess.

        Args:
            image: The image to resize.
            width: The width to resize the image to.
            height: The height to resize the image to.
        """
    ratio = width / height
    src_ratio = image.width / image.height
    src_w = (width if ratio > src_ratio else image.width * height // image.
        height)
    src_h = (height if ratio <= src_ratio else image.height * width //
        image.width)
    resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[
        'lanczos'])
    res = Image.new('RGB', (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    return res
