def load_coco_panoptic_json(json_file, image_dir, gt_dir, meta,
    dataset_name_in_dict='coco'):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/coco/train2017".
        gt_dir (str): path to the raw annotations. e.g., "~/coco/panoptic_train2017".
        json_file (str): path to the json file. e.g., "~/coco/annotations/panoptic_train2017.json".

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info['category_id'] in meta[
            'thing_dataset_id_to_contiguous_id']:
            segment_info['category_id'] = meta[
                'thing_dataset_id_to_contiguous_id'][segment_info[
                'category_id']]
            segment_info['isthing'] = True
        else:
            segment_info['category_id'] = meta[
                'stuff_dataset_id_to_contiguous_id'][segment_info[
                'category_id']]
            segment_info['isthing'] = False
        return segment_info
    with PathManager.open(json_file) as f:
        json_info = json.load(f)
    ret = []
    for ann in json_info['annotations']:
        image_id = int(ann['image_id'])
        image_file = os.path.join(image_dir, os.path.splitext(ann[
            'file_name'])[0] + '.jpg')
        label_file = os.path.join(gt_dir, ann['file_name'])
        segments_info = [_convert_category_id(x, meta) for x in ann[
            'segments_info']]
        record = {'file_name': image_file, 'image_id': image_id,
            'pan_seg_file_name': label_file, 'segments_info': segments_info}
        record['task'] = 'detection'
        record['has_stuff'] = True
        record['dataset_name'] = dataset_name_in_dict
        ret.append({'file_name': image_file, 'image_id': image_id,
            'pan_seg_file_name': label_file, 'segments_info': segments_info})
    assert len(ret), f'No images found in {image_dir}!'
    assert PathManager.isfile(ret[0]['file_name']), ret[0]['file_name']
    assert PathManager.isfile(ret[0]['pan_seg_file_name']), ret[0][
        'pan_seg_file_name']
    return ret
