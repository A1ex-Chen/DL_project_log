def register_lvis_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda : load_lvis_json(json_file,
        image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file, image_root=
        image_root, evaluator_type='lvis', **metadata)
