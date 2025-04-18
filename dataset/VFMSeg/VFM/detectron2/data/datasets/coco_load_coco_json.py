def load_coco_json(json_file, image_root, dataset_name=None,
    extra_annotation_keys=None, dataset_name_in_dict='coco', segm_only=False):
    """
    Load a json file with COCO's instances annotation format.
    Currently supports instance detection, instance segmentation,
    and person keypoints annotations.

    Args:
        json_file (str): full path to the json file in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.
        extra_annotation_keys (list[str]): list of per-annotation keys that should also be
            loaded into the dataset dict (besides "iscrowd", "bbox", "keypoints",
            "category_id", "segmentation"). The values for these keys will be returned as-is.
            For example, the densepose annotations are loaded in this way.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    """
    from pycocotools.coco import COCO
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info('Loading {} takes {:.2f} seconds.'.format(json_file,
            timer.seconds()))
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c['name'] for c in sorted(cats, key=lambda x: x['id'])
            ]
        meta.thing_classes = thing_classes
        if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
            if 'coco' not in dataset_name:
                logger.warning(
                    """
Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
"""
                    )
        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f'{json_file} contains {total_num_anns} annotations, but only {total_num_valid_anns} of them match to images in the file.'
            )
    if 'minival' not in json_file:
        ann_ids = [ann['id'] for anns_per_image in anns for ann in
            anns_per_image]
        assert len(set(ann_ids)) == len(ann_ids
            ), "Annotation ids in '{}' are not unique!".format(json_file)
    imgs_anns = list(zip(imgs, anns))
    logger.info('Loaded {} images in COCO format from {}'.format(len(
        imgs_anns), json_file))
    dataset_dicts = []
    ann_keys = ['iscrowd', 'bbox', 'keypoints', 'category_id'] + (
        extra_annotation_keys or [])
    num_instances_without_valid_segmentation = 0
    for img_dict, anno_dict_list in imgs_anns:
        record = {}
        record['file_name'] = os.path.join(image_root, img_dict['file_name'])
        record['height'] = img_dict['height']
        record['width'] = img_dict['width']
        image_id = record['image_id'] = img_dict['id']
        objs = []
        for anno in anno_dict_list:
            assert anno['image_id'] == image_id
            assert anno.get('ignore', 0
                ) == 0, '"ignore" in COCO json file is not supported.'
            obj = {key: anno[key] for key in ann_keys if key in anno}
            if 'bbox' in obj and len(obj['bbox']) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! This json does not have valid COCO format."
                    )
            segm = anno.get('segmentation', None)
            if segm:
                if isinstance(segm, dict):
                    if isinstance(segm['counts'], list):
                        segm = mask_util.frPyObjects(segm, *segm['size'])
                else:
                    segm = [poly for poly in segm if len(poly) % 2 == 0 and
                        len(poly) >= 6]
                    if len(segm) == 0:
                        num_instances_without_valid_segmentation += 1
                        continue
                obj['segmentation'] = segm
            elif segm_only:
                num_instances_without_valid_segmentation += 1
                continue
            keypts = anno.get('keypoints', None)
            if keypts:
                for idx, v in enumerate(keypts):
                    if idx % 3 != 2:
                        keypts[idx] = v + 0.5
                obj['keypoints'] = keypts
            obj['bbox_mode'] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj['category_id']
                try:
                    obj['category_id'] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} but this id does not exist in 'categories' of the json file."
                        ) from e
            objs.append(obj)
        record['annotations'] = objs
        record['task'] = 'detection'
        record['dataset_name'] = dataset_name_in_dict
        dataset_dicts.append(record)
    if num_instances_without_valid_segmentation > 0:
        logger.warning(
            'Filtered out {} instances without valid segmentation. '.format
            (num_instances_without_valid_segmentation) +
            'There might be issues in your dataset generation process.  Please check https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html carefully'
            )
    return dataset_dicts
