def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(key, _get_builtin_metadata(dataset_name
                ), os.path.join(root, json_file) if '://' not in json_file else
                json_file, os.path.join(root, image_root))
    for prefix, (panoptic_root, panoptic_json, semantic_root
        ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[:-len('_panoptic')]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = (instances_meta.image_root,
            instances_meta.json_file)
        register_coco_panoptic_separated(prefix, _get_builtin_metadata(
            'coco_panoptic_separated'), image_root, os.path.join(root,
            panoptic_root), os.path.join(root, panoptic_json), os.path.join
            (root, semantic_root), instances_json)
        register_coco_panoptic(prefix, _get_builtin_metadata(
            'coco_panoptic_standard'), image_root, os.path.join(root,
            panoptic_root), os.path.join(root, panoptic_json), instances_json)
