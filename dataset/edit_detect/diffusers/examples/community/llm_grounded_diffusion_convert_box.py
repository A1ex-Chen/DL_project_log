@classmethod
def convert_box(cls, box, height, width):
    x_min, y_min = box[0] / width, box[1] / height
    w_box, h_box = box[2] / width, box[3] / height
    x_max, y_max = x_min + w_box, y_min + h_box
    return x_min, y_min, x_max, y_max
