def get_new_h_w(h, w, scale_factor=8):
    new_h = h // scale_factor ** 2
    if h % scale_factor ** 2 != 0:
        new_h += 1
    new_w = w // scale_factor ** 2
    if w % scale_factor ** 2 != 0:
        new_w += 1
    return new_h * scale_factor, new_w * scale_factor
