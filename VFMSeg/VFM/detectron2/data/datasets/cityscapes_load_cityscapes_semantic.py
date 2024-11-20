def load_cityscapes_semantic(image_dir, gt_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    gt_dir = PathManager.get_local_path(gt_dir)
    for image_file, _, label_file, json_file in _get_cityscapes_files(image_dir
        , gt_dir):
        label_file = label_file.replace('labelIds', 'labelTrainIds')
        with PathManager.open(json_file, 'r') as f:
            jsonobj = json.load(f)
        ret.append({'file_name': image_file, 'sem_seg_file_name':
            label_file, 'height': jsonobj['imgHeight'], 'width': jsonobj[
            'imgWidth']})
    assert len(ret), f'No images found in {image_dir}!'
    assert PathManager.isfile(ret[0]['sem_seg_file_name']
        ), 'Please generate labelTrainIds.png with cityscapesscripts/preparation/createTrainIdLabelImgs.py'
    return ret
