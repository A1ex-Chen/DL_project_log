def exif_size(img):
    """Returns exif-corrected PIL size."""
    s = img.size
    with contextlib.suppress(Exception):
        rotation = dict(img._getexif().items())[orientation]
        if rotation in [6, 8]:
            s = s[1], s[0]
    return s
