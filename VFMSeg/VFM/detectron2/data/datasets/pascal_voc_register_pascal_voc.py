def register_pascal_voc(name, dirname, split, year, class_names=CLASS_NAMES):
    DatasetCatalog.register(name, lambda : load_voc_instances(dirname,
        split, class_names))
    MetadataCatalog.get(name).set(thing_classes=list(class_names), dirname=
        dirname, year=year, split=split)
