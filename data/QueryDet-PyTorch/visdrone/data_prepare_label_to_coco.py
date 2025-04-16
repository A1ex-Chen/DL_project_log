def label_to_coco(label, id, img_id):
    x, y, w, h = label['bbox']
    new_label = {'category_id': label['class'], 'id': id, 'iscrowd': 0,
        'image_id': img_id, 'area': w * h, 'segmentation': [], 'bbox': [x,
        y, w, h]}
    return new_label
