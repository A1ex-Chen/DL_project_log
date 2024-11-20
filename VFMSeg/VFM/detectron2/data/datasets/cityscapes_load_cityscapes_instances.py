def load_cityscapes_instances(image_dir, gt_dir, from_json=True,
    to_polygons=True):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".
        from_json (bool): whether to read annotations from the raw json file or the png files.
        to_polygons (bool): whether to represent the segmentation as polygons
            (COCO's format) instead of masks (cityscapes's format).

    Returns:
        list[dict]: a list of dicts in Detectron2 standard format. (See
        `Using Custom Datasets </tutorials/datasets.html>`_ )
    """
    if from_json:
        assert to_polygons, "Cityscapes's json annotations are in polygon format. Converting to mask format is not supported now."
    files = _get_cityscapes_files(image_dir, gt_dir)
    logger.info('Preprocessing cityscapes annotations ...')
    pool = mp.Pool(processes=max(mp.cpu_count() // get_world_size() // 2, 4))
    ret = pool.map(functools.partial(_cityscapes_files_to_dict, from_json=
        from_json, to_polygons=to_polygons), files)
    logger.info('Loaded {} images from {}'.format(len(ret), image_dir))
    from cityscapesscripts.helpers.labels import labels
    labels = [l for l in labels if l.hasInstances and not l.ignoreInEval]
    dataset_id_to_contiguous_id = {l.id: idx for idx, l in enumerate(labels)}
    for dict_per_image in ret:
        for anno in dict_per_image['annotations']:
            anno['category_id'] = dataset_id_to_contiguous_id[anno[
                'category_id']]
    return ret
