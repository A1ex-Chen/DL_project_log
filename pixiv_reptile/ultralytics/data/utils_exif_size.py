def exif_size(img: Image.Image):
    """Returns exif-corrected PIL size."""
    s = img.size
    if img.format == 'JPEG':
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)
                if rotation in {6, 8}:
                    s = s[1], s[0]
    return s
