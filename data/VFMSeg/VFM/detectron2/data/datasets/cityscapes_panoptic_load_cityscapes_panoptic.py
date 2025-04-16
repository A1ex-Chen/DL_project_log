def load_cityscapes_panoptic(image_dir, gt_dir, gt_json, meta):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file. e.g.,
            "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
        meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id"
            and "stuff_dataset_id_to_contiguous_id" to map category ids to
            contiguous ids for training.

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
        else:
            segment_info['category_id'] = meta[
                'stuff_dataset_id_to_contiguous_id'][segment_info[
                'category_id']]
        return segment_info
    assert os.path.exists(gt_json
        ), 'Please run `python cityscapesscripts/preparation/createPanopticImgs.py` to generate label files.'
    with open(gt_json) as f:
        json_info = json.load(f)
    files = get_cityscapes_panoptic_files(image_dir, gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info in files:
        sem_label_file = image_file.replace('leftImg8bit', 'gtFine').split('.'
            )[0] + '_labelTrainIds.png'
        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append({'file_name': image_file, 'image_id': '_'.join(os.path.
            splitext(os.path.basename(image_file))[0].split('_')[:3]),
            'sem_seg_file_name': sem_label_file, 'pan_seg_file_name':
            label_file, 'segments_info': segments_info})
    assert len(ret), f'No images found in {image_dir}!'
    assert PathManager.isfile(ret[0]['sem_seg_file_name']
        ), 'Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py'
    assert PathManager.isfile(ret[0]['pan_seg_file_name']
        ), 'Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py'
    return ret
