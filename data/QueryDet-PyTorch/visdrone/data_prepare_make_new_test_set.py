def make_new_test_set(img_root, label_root, new_img_root, new_label_json):
    all_labels = utils.read_all_labels(label_root)
    annotations = []
    images = []
    ann_id = 0
    img_id = 0
    for filename, labels in tqdm(all_labels.items()):
        img_path = filename.replace('txt', 'jpg')
        h, w = copy_image(img_root, img_path, new_img_root)
        images.append({'file_name': img_path, 'height': h, 'width': w, 'id':
            img_id})
        for label in labels:
            coco_label = label_to_coco(label, ann_id, img_id)
            if coco_label != None:
                ann_id += 1
                annotations.append(coco_label)
        img_id += 1
    make_json(images, annotations, new_label_json)
