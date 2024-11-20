def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.

    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`

    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527

    Args:
        image (PIL.Image): a PIL image

    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, 'getexif'):
        return image
    try:
        exif = image.getexif()
    except Exception:
        exif = None
    if exif is None:
        return image
    orientation = exif.get(_EXIF_ORIENT)
    method = {(2): Image.FLIP_LEFT_RIGHT, (3): Image.ROTATE_180, (4): Image
        .FLIP_TOP_BOTTOM, (5): Image.TRANSPOSE, (6): Image.ROTATE_270, (7):
        Image.TRANSVERSE, (8): Image.ROTATE_90}.get(orientation)
    if method is not None:
        return image.transpose(method)
    return image
