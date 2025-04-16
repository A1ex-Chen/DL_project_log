def point_xy_expand2square(point, *, w, h):
    pseudo_box = point[0], point[1], point[0], point[1]
    expanded_box = box_xyxy_expand2square(box=pseudo_box, w=w, h=h)
    expanded_point = expanded_box[0], expanded_box[1]
    return expanded_point
