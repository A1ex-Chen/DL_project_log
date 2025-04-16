def _resize_and_fill(self, image: PIL.Image.Image, width: int, height: int
    ) ->PIL.Image.Image:
    """
        Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center
        the image within the dimensions, filling empty with data from image.

        Args:
            image: The image to resize.
            width: The width to resize the image to.
            height: The height to resize the image to.
        """
    ratio = width / height
    src_ratio = image.width / image.height
    src_w = (width if ratio < src_ratio else image.width * height // image.
        height)
    src_h = (height if ratio >= src_ratio else image.height * width //
        image.width)
    resized = image.resize((src_w, src_h), resample=PIL_INTERPOLATION[
        'lanczos'])
    res = Image.new('RGB', (width, height))
    res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    if ratio < src_ratio:
        fill_height = height // 2 - src_h // 2
        if fill_height > 0:
            res.paste(resized.resize((width, fill_height), box=(0, 0, width,
                0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.
                height, width, resized.height)), box=(0, fill_height + src_h))
    elif ratio > src_ratio:
        fill_width = width // 2 - src_w // 2
        if fill_width > 0:
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0,
                height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.
                width, 0, resized.width, height)), box=(fill_width + src_w, 0))
    return res
