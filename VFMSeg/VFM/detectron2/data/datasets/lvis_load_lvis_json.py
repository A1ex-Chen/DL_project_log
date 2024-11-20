def load_lvis_json(json_file, image_root, dataset_name=None,
    extra_annotation_keys=None):
    """
    Load a json file in LVIS's annotation format.

    Args:
        json_file (str): full path to the LVIS json annotation file.
        image_root (str): the directory where the images in this json file exists.
        dataset_name (str): the name of the dataset (e.g., "lvis_v0.5_train").
            If provided, this function will put "thing_classes" into the metadata
            associated with this dataset.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "bbox", "bbox_mode", "category_id",
            "segmentation"). The values for these keys will be returned as-is.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from lvis import LVIS
    json_file = PathManager.get_local_path(json_file)
    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info('Loading {} takes {:.2f} seconds.'.format(json_file,
            timer.seconds()))
    if dataset_name is not None:
        meta = get_lvis_instances_meta(dataset_name)
        MetadataCatalog.get(dataset_name).set(**meta)
    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]
    ann_ids = [ann['id'] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids
        ), "Annotation ids in '{}' are not unique".format(json_file)
    imgs_anns = list(zip(imgs, anns))
    logger.info('Loaded {} images in the LVIS format from {}'.format(len(
        imgs_anns), json_file))
    if extra_annotation_keys:
        logger.info('The following extra annotation keys will be loaded: {} '
            .format(extra_annotation_keys))
    else:
        extra_annotation_keys = []

    def get_file_name(img_root, img_dict):
        split_folder, file_name = img_dict['coco_url'].split('/')[-2:]
        return os.path.join(img_root + split_folder, file_name)
    dataset_dicts = []
    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record['file_name'] = get_file_name(image_root, img_dict)
        record['height'] = img_dict['height']
        record['width'] = img_dict['width']
        record['not_exhaustive_category_ids'] = img_dict.get(
            'not_exhaustive_category_ids', [])
        record['neg_category_ids'] = img_dict.get('neg_category_ids', [])
        image_id = record['image_id'] = img_dict['id']
        objs = []
        for anno in anno_dict_list:
            assert anno['image_id'] == image_id
            obj = {'bbox': anno['bbox'], 'bbox_mode': BoxMode.XYWH_ABS}
            if (dataset_name is not None and 
                'thing_dataset_id_to_contiguous_id' in meta):
                obj['category_id'] = meta['thing_dataset_id_to_contiguous_id'][
                    anno['category_id']]
            else:
                obj['category_id'] = anno['category_id'] - 1
            segm = anno['segmentation']
            valid_segm = [poly for poly in segm if len(poly) % 2 == 0 and 
                len(poly) >= 6]
            assert len(segm) == len(valid_segm
                ), 'Annotation contains an invalid polygon with < 3 points'
            assert len(segm) > 0
            obj['segmentation'] = segm
            for extra_ann_key in extra_annotation_keys:
                obj[extra_ann_key] = anno[extra_ann_key]
            objs.append(obj)
        record['annotations'] = objs
        dataset_dicts.append(record)
    return dataset_dicts
