def norm_box_xyxy(box, *, w, h):
    x1, y1, x2, y2 = box
    norm_x1 = max(0.0, min(x1 / w, 1.0))
    norm_y1 = max(0.0, min(y1 / h, 1.0))
    norm_x2 = max(0.0, min(x2 / w, 1.0))
    norm_y2 = max(0.0, min(y2 / h, 1.0))
    normalized_box = round(norm_x1, 3), round(norm_y1, 3), round(norm_x2, 3
        ), round(norm_y2, 3)
    return normalized_box
