def load_voc_instances(dirname: str, split: str, class_names: Union[List[
    str], Tuple[str, ...]]):
    """
    Load Pascal VOC detection annotations to Detectron2 format.

    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
        class_names: list or tuple of class names
    """
    with PathManager.open(os.path.join(dirname, 'ImageSets', 'Main', split +
        '.txt')) as f:
        fileids = np.loadtxt(f, dtype=np.str)
    annotation_dirname = PathManager.get_local_path(os.path.join(dirname,
        'Annotations/'))
    dicts = []
    for fileid in fileids:
        anno_file = os.path.join(annotation_dirname, fileid + '.xml')
        jpeg_file = os.path.join(dirname, 'JPEGImages', fileid + '.jpg')
        with PathManager.open(anno_file) as f:
            tree = ET.parse(f)
        r = {'file_name': jpeg_file, 'image_id': fileid, 'height': int(tree
            .findall('./size/height')[0].text), 'width': int(tree.findall(
            './size/width')[0].text)}
        instances = []
        for obj in tree.findall('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            bbox = [float(bbox.find(x).text) for x in ['xmin', 'ymin',
                'xmax', 'ymax']]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instances.append({'category_id': class_names.index(cls), 'bbox':
                bbox, 'bbox_mode': BoxMode.XYXY_ABS})
        r['annotations'] = instances
        dicts.append(r)
    return dicts
