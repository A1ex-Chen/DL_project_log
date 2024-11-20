def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata('cityscapes')
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        inst_key = key.format(task='instance_seg')
        DatasetCatalog.register(inst_key, lambda x=image_dir, y=gt_dir:
            load_cityscapes_instances(x, y, from_json=True, to_polygons=True))
        MetadataCatalog.get(inst_key).set(image_dir=image_dir, gt_dir=
            gt_dir, evaluator_type='cityscapes_instance', **meta)
        sem_key = key.format(task='sem_seg')
        DatasetCatalog.register(sem_key, lambda x=image_dir, y=gt_dir:
            load_cityscapes_semantic(x, y))
        MetadataCatalog.get(sem_key).set(image_dir=image_dir, gt_dir=gt_dir,
            evaluator_type='cityscapes_sem_seg', ignore_label=255, **meta)
