def register_all_pascal_voc(root):
    SPLITS = [('voc_2007_trainval', 'VOC2007', 'trainval'), (
        'voc_2007_train', 'VOC2007', 'train'), ('voc_2007_val', 'VOC2007',
        'val'), ('voc_2007_test', 'VOC2007', 'test'), ('voc_2012_trainval',
        'VOC2012', 'trainval'), ('voc_2012_train', 'VOC2012', 'train'), (
        'voc_2012_val', 'VOC2012', 'val')]
    for name, dirname, split in SPLITS:
        year = 2007 if '2007' in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = 'pascal_voc'
