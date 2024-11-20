def exif_size(img):
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = s[1], s[0]
        elif rotation == 8:
            s = s[1], s[0]
    except:
        pass
    return s
