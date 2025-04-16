def clipped_zoom(img, zoom_factor):
    h, w = img.shape[0], img.shape[1]
    ch = int(np.ceil(h / zoom_factor))
    chw = int(np.ceil(w / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + chw], (zoom_factor,
        zoom_factor, 1), order=1)
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + w]
