def convert_box(size, box):
    dw, dh = 1.0 / size[0], 1.0 / size[1]
    x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1, box[
        1] - box[0], box[3] - box[2]
    return x * dw, y * dh, w * dw, h * dh
