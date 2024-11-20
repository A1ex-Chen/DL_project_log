def compress_one_image(f, f_new=None, max_dim=1920, quality=50):
    """
    Compresses a single image file to reduced size while preserving its aspect ratio and quality using either the Python
    Imaging Library (PIL) or OpenCV library. If the input image is smaller than the maximum dimension, it will not be
    resized.

    Args:
        f (str): The path to the input image file.
        f_new (str, optional): The path to the output image file. If not specified, the input file will be overwritten.
        max_dim (int, optional): The maximum dimension (width or height) of the output image. Default is 1920 pixels.
        quality (int, optional): The image compression quality as a percentage. Default is 50%.

    Example:
        ```python
        from pathlib import Path
        from ultralytics.data.utils import compress_one_image

        for f in Path('path/to/dataset').rglob('*.jpg'):
            compress_one_image(f)
        ```
    """
    try:
        im = Image.open(f)
        r = max_dim / max(im.height, im.width)
        if r < 1.0:
            im = im.resize((int(im.width * r), int(im.height * r)))
        im.save(f_new or f, 'JPEG', quality=quality, optimize=True)
    except Exception as e:
        LOGGER.info(f'WARNING ⚠️ HUB ops PIL failure {f}: {e}')
        im = cv2.imread(f)
        im_height, im_width = im.shape[:2]
        r = max_dim / max(im_height, im_width)
        if r < 1.0:
            im = cv2.resize(im, (int(im_width * r), int(im_height * r)),
                interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(f_new or f), im)
