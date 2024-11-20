def resize(images: PIL.Image.Image, img_size: int) ->PIL.Image.Image:
    w, h = images.size
    coef = w / h
    w, h = img_size, img_size
    if coef >= 1:
        w = int(round(img_size / 8 * coef) * 8)
    else:
        h = int(round(img_size / 8 / coef) * 8)
    images = images.resize((w, h), resample=PIL_INTERPOLATION['bicubic'],
        reducing_gap=None)
    return images
