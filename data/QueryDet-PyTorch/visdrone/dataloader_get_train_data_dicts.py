def get_train_data_dicts(json_file, img_root, filter_empty=False):
    data = json.load(open(json_file))
    images = {x['id']: {'file': x['file_name'], 'height': x['height'],
        'width': x['width']} for x in data['images']}
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations.keys():
            annotations[img_id] = []
        annotations[img_id].append({'bbox': ann['bbox'], 'category_id': ann
            ['category_id'] - 1, 'iscrowd': ann['iscrowd'], 'area': ann[
            'area']})
    for img_id in images.keys():
        if img_id not in annotations.keys():
            annotations[img_id] = []
    data_dicts = []
    for img_id in images.keys():
        if filter_empty and len(annotations[img_id]) == 0:
            continue
        data_dict = {}
        data_dict['file_name'] = str(os.path.join(img_root, images[img_id][
            'file']))
        data_dict['height'] = images[img_id]['height']
        data_dict['width'] = images[img_id]['width']
        data_dict['image_id'] = img_id
        data_dict['annotations'] = []
        for ann in annotations[img_id]:
            data_dict['annotations'].append({'bbox': ann['bbox'], 'iscrowd':
                ann['iscrowd'], 'category_id': ann['category_id'],
                'bbox_mode': BoxMode.XYWH_ABS})
        data_dicts.append(data_dict)
    return data_dicts
