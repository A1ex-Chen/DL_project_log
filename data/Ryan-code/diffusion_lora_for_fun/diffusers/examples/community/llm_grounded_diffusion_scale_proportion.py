def scale_proportion(obj_box, H, W):
    x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
    box_w, box_h = round((obj_box[2] - obj_box[0]) * W), round((obj_box[3] -
        obj_box[1]) * H)
    x_max, y_max = x_min + box_w, y_min + box_h
    x_min, y_min = max(x_min, 0), max(y_min, 0)
    x_max, y_max = min(x_max, W), min(y_max, H)
    return x_min, y_min, x_max, y_max
