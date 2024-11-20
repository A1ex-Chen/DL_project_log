def register_coco_panoptic_separated(name, metadata, image_root,
    panoptic_root, panoptic_json, sem_seg_root, instances_json):
    """
    Register a "separated" version of COCO panoptic segmentation dataset named `name`.
    The annotations in this registered dataset will contain both instance annotations and
    semantic annotations, each with its own contiguous ids. Hence it's called "separated".

    It follows the setting used by the PanopticFPN paper:

    1. The instance annotations directly come from polygons in the COCO
       instances annotation task, rather than from the masks in the COCO panoptic annotations.

       The two format have small differences:
       Polygons in the instance annotations may have overlaps.
       The mask annotations are produced by labeling the overlapped polygons
       with depth ordering.

    2. The semantic annotations are converted from panoptic annotations, where
       all "things" are assigned a semantic id of 0.
       All semantic categories will therefore have ids in contiguous
       range [1, #stuff_categories].

    This function will also register a pure semantic segmentation dataset
    named ``name + '_stuffonly'``.

    Args:
        name (str): the name that identifies a dataset,
            e.g. "coco_2017_train_panoptic"
        metadata (dict): extra metadata associated with this dataset.
        image_root (str): directory which contains all the images
        panoptic_root (str): directory which contains panoptic annotation images
        panoptic_json (str): path to the json panoptic annotation file
        sem_seg_root (str): directory which contains all the ground truth segmentation annotations.
        instances_json (str): path to the json instance annotation file
    """
    panoptic_name = name + '_separated'
    DatasetCatalog.register(panoptic_name, lambda : merge_to_panoptic(
        load_coco_json(instances_json, image_root, panoptic_name),
        load_sem_seg(sem_seg_root, image_root)))
    MetadataCatalog.get(panoptic_name).set(panoptic_root=panoptic_root,
        image_root=image_root, panoptic_json=panoptic_json, sem_seg_root=
        sem_seg_root, json_file=instances_json, evaluator_type=
        'coco_panoptic_seg', ignore_label=255, **metadata)
    semantic_name = name + '_stuffonly'
    DatasetCatalog.register(semantic_name, lambda : load_sem_seg(
        sem_seg_root, image_root))
    MetadataCatalog.get(semantic_name).set(sem_seg_root=sem_seg_root,
        image_root=image_root, evaluator_type='sem_seg', ignore_label=255,
        **metadata)
