def register_coco_instances(name, metadata, json_file, image_root,
    dataset_name_in_dict='coco', segm_only=False):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    DatasetCatalog.register(name, lambda : load_coco_json(json_file,
        image_root, name, dataset_name_in_dict=dataset_name_in_dict,
        segm_only=segm_only))
    MetadataCatalog.get(name).set(json_file=json_file, image_root=
        image_root, evaluator_type='coco', **metadata)
