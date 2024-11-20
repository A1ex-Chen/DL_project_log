def get_new_label(label, img_path, cy, cx, id, img_id_base):
    if label['class'] == 0 or label['ignore']:
        return None
    x, y, w, h = label['bbox']
    if x < cx and y < cy:
        nx = x
        ny = y
        nw = min(x + w, cx) - x
        nh = min(y + h, cy) - y
        img_id = img_id_base
    elif x < cx and y >= cy:
        nx = x
        ny = y - cy
        nw = min(x + w, cx) - x
        nh = h
        img_id = img_id_base + 2
    elif x >= cx and y < cy:
        nx = x - cx
        ny = y
        nw = w
        nh = min(y + h, cy) - y
        img_id = img_id_base + 1
    else:
        nx = x - cx
        ny = y - cy
        nw = w
        nh = h
        img_id = img_id_base + 3
    new_label = {'category_id': label['class'], 'id': id, 'iscrowd': 0,
        'image_id': img_id, 'area': nw * nh, 'segmentation': [], 'bbox': [
        nx, ny, nw, nh]}
    return new_label
