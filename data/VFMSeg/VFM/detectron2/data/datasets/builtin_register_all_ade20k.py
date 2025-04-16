def register_all_ade20k(root):
    root = os.path.join(root, 'ADEChallengeData2016')
    for name, dirname in [('train', 'training'), ('val', 'validation')]:
        image_dir = os.path.join(root, 'images', dirname)
        gt_dir = os.path.join(root, 'annotations_detectron2', dirname)
        name = f'ade20k_sem_seg_{name}'
        DatasetCatalog.register(name, lambda x=image_dir, y=gt_dir:
            load_sem_seg(y, x, gt_ext='png', image_ext='jpg'))
        MetadataCatalog.get(name).set(stuff_classes=
            ADE20K_SEM_SEG_CATEGORIES[:], image_root=image_dir,
            sem_seg_root=gt_dir, evaluator_type='sem_seg', ignore_label=255)
