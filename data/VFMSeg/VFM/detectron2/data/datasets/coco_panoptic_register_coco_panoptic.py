def register_coco_panoptic(name, metadata, image_root, panoptic_root,
    panoptic_json, instances_json=None):
    """
    Register a "standard" version of COCO panoptic segmentation dataset named `name`.
    The dictionaries in this registered dataset follows detectron2's standard format.
    Hence it's called "standard".

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images in COCO format
        panoptic_json (str): path to the json panoptic annotation file in COCO format
        sem_seg_root (none): not used, to be consistent with
            `register_coco_panoptic_separated`.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name
    DatasetCatalog.register(panoptic_name, lambda : load_coco_panoptic_json
        (panoptic_json, image_root, panoptic_root, metadata))
    MetadataCatalog.get(panoptic_name).set(panoptic_root=panoptic_root,
        image_root=image_root, panoptic_json=panoptic_json, json_file=
        instances_json, evaluator_type='coco_panoptic_seg', ignore_label=
        255, label_divisor=1000, **metadata)
