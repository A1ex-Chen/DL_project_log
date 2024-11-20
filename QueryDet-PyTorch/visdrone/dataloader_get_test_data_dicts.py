def get_test_data_dicts(json_file, img_root):
    data = json.load(open(json_file))
    images = {x['id']: {'file': x['file_name'], 'height': x['height'],
        'width': x['width']} for x in data['images']}
    data_dicts = []
    for img_id in images.keys():
        data_dict = {}
        data_dict['file_name'] = str(os.path.join(img_root, images[img_id][
            'file']))
        data_dict['height'] = images[img_id]['height']
        data_dict['width'] = images[img_id]['width']
        data_dict['image_id'] = img_id
        data_dict['annotations'] = []
        data_dicts.append(data_dict)
    return data_dicts
