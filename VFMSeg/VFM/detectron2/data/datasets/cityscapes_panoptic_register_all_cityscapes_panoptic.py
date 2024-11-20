def register_all_cityscapes_panoptic(root):
    meta = {}
    thing_classes = [k['name'] for k in CITYSCAPES_CATEGORIES]
    thing_colors = [k['color'] for k in CITYSCAPES_CATEGORIES]
    stuff_classes = [k['name'] for k in CITYSCAPES_CATEGORIES]
    stuff_colors = [k['color'] for k in CITYSCAPES_CATEGORIES]
    meta['thing_classes'] = thing_classes
    meta['thing_colors'] = thing_colors
    meta['stuff_classes'] = stuff_classes
    meta['stuff_colors'] = stuff_colors
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}
    for k in CITYSCAPES_CATEGORIES:
        if k['isthing'] == 1:
            thing_dataset_id_to_contiguous_id[k['id']] = k['trainId']
        else:
            stuff_dataset_id_to_contiguous_id[k['id']] = k['trainId']
    meta['thing_dataset_id_to_contiguous_id'
        ] = thing_dataset_id_to_contiguous_id
    meta['stuff_dataset_id_to_contiguous_id'
        ] = stuff_dataset_id_to_contiguous_id
    for key, (image_dir, gt_dir, gt_json
        ) in _RAW_CITYSCAPES_PANOPTIC_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)
        DatasetCatalog.register(key, lambda x=image_dir, y=gt_dir, z=
            gt_json: load_cityscapes_panoptic(x, y, z, meta))
        MetadataCatalog.get(key).set(panoptic_root=gt_dir, image_root=
            image_dir, panoptic_json=gt_json, gt_dir=gt_dir.replace(
            'cityscapes_panoptic_', ''), evaluator_type=
            'cityscapes_panoptic_seg', ignore_label=255, label_divisor=1000,
            **meta)
