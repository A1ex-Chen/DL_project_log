def get_lvis_instances_meta(dataset_name):
    """
    Load LVIS metadata.

    Args:
        dataset_name (str): LVIS dataset name without the split name (e.g., "lvis_v0.5").

    Returns:
        dict: LVIS metadata with keys: thing_classes
    """
    if 'cocofied' in dataset_name:
        return _get_coco_instances_meta()
    if 'v0.5' in dataset_name:
        return _get_lvis_instances_meta_v0_5()
    elif 'v1' in dataset_name:
        return _get_lvis_instances_meta_v1()
    raise ValueError('No built-in metadata for dataset {}'.format(dataset_name)
        )
