def filter_images_with_few_keypoints(dataset_dicts, min_keypoints_per_image):
    """
    Filter out images with too few number of keypoints.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.

    Returns:
        list[dict]: the same format as dataset_dicts, but filtered.
    """
    num_before = len(dataset_dicts)

    def visible_keypoints_in_image(dic):
        annotations = dic['annotations']
        return sum((np.array(ann['keypoints'][2::3]) > 0).sum() for ann in
            annotations if 'keypoints' in ann)
    dataset_dicts = [x for x in dataset_dicts if visible_keypoints_in_image
        (x) >= min_keypoints_per_image]
    num_after = len(dataset_dicts)
    logger = logging.getLogger(__name__)
    logger.info('Removed {} images with fewer than {} keypoints.'.format(
        num_before - num_after, min_keypoints_per_image))
    return dataset_dicts
