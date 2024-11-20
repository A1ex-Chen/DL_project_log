def filter_images_with_only_crowd_annotations(dataset_dicts):
    """
    Filter out images with none annotations or only crowd annotations
    (i.e., images without non-crowd annotations).
    A common training-time preprocessing on COCO dataset.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format, but filtered.
    """
    num_before = len(dataset_dicts)

    def valid(anns):
        for ann in anns:
            if ann.get('iscrowd', 0) == 0:
                return True
        return False
    dataset_dicts = [x for x in dataset_dicts if valid(x['annotations'])]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info('Removed {} images with no usable annotations. {} images left.'
        .format(num_before - num_after, num_after))
    return dataset_dicts
