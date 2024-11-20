def make_new_train_set(img_root, label_root, new_img_root, new_label_json):
    all_labels = utils.read_all_labels(label_root)
    annotations = []
    images = []
    ann_id = 0
    img_id = 0
    for filename, labels in tqdm(all_labels.items()):
        img_path = filename.replace('txt', 'jpg')
        h, w, cy, cx = crop_and_save_image(img_root, img_path, new_img_root)
        images.append({'file_name': get_save_path(img_path, 0), 'height':
            cy, 'width': cx, 'id': img_id})
        images.append({'file_name': get_save_path(img_path, 1), 'height':
            cy, 'width': w - cx, 'id': img_id + 1})
        images.append({'file_name': get_save_path(img_path, 2), 'height': h -
            cy, 'width': cx, 'id': img_id + 2})
        images.append({'file_name': get_save_path(img_path, 3), 'height': h -
            cy, 'width': w - cx, 'id': img_id + 3})
        for label in labels:
            new_label = get_new_label(label, img_path, cy, cx, ann_id, img_id)
            if new_label != None:
                ann_id += 1
                annotations.append(new_label)
        img_id += 4
    make_json(images, annotations, new_label_json)
