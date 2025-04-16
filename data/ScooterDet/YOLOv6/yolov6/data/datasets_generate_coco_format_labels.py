@staticmethod
def generate_coco_format_labels(img_info, class_names, save_path):
    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, class_name in enumerate(class_names):
        dataset['categories'].append({'id': i, 'name': class_name,
            'supercategory': ''})
    ann_id = 0
    LOGGER.info(f'Convert to COCO format')
    for i, (img_path, info) in enumerate(tqdm(img_info.items())):
        labels = info['labels'] if info['labels'] else []
        img_id = osp.splitext(osp.basename(img_path))[0]
        img_h, img_w = info['shape']
        dataset['images'].append({'file_name': os.path.basename(img_path),
            'id': img_id, 'width': img_w, 'height': img_h})
        if labels:
            for label in labels:
                c, x, y, w, h = label[:5]
                x1 = (x - w / 2) * img_w
                y1 = (y - h / 2) * img_h
                x2 = (x + w / 2) * img_w
                y2 = (y + h / 2) * img_h
                cls_id = int(c)
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                dataset['annotations'].append({'area': h * w, 'bbox': [x1,
                    y1, w, h], 'category_id': cls_id, 'id': ann_id,
                    'image_id': img_id, 'iscrowd': 0, 'segmentation': []})
                ann_id += 1
    with open(save_path, 'w') as f:
        json.dump(dataset, f)
        LOGGER.info(
            f'Convert to COCO format finished. Resutls saved in {save_path}')
