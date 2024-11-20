def load_sem_seg(gt_root, image_root, gt_ext='png', image_ext='jpg',
    dataset_name='sem_seg'):
    """
    Load semantic segmentation datasets. All files under "gt_root" with "gt_ext" extension are
    treated as ground truth annotations and all files under "image_root" with "image_ext" extension
    as input images. Ground truth and input images are matched using file paths relative to
    "gt_root" and "image_root" respectively without taking into account file extensions.
    This works for COCO as well as some other datasets.

    Args:
        gt_root (str): full path to ground truth semantic segmentation files. Semantic segmentation
            annotations are stored as images with integer values in pixels that represent
            corresponding semantic labels.
        image_root (str): the directory where the input images are.
        gt_ext (str): file extension for ground truth annotations.
        image_ext (str): file extension for input images.

    Returns:
        list[dict]:
            a list of dicts in detectron2 standard format without instance-level
            annotation.

    Notes:
        1. This function does not read the image and ground truth files.
           The results do not have the "image" and "sem_seg" fields.
    """

    def file2id(folder_path, file_path):
        image_id = os.path.normpath(os.path.relpath(file_path, start=
            folder_path))
        image_id = os.path.splitext(image_id)[0]
        return image_id
    input_files = sorted((os.path.join(image_root, f) for f in PathManager.
        ls(image_root) if f.endswith(image_ext)), key=lambda file_path:
        file2id(image_root, file_path))
    gt_files = sorted((os.path.join(gt_root, f) for f in PathManager.ls(
        gt_root) if f.endswith(gt_ext)), key=lambda file_path: file2id(
        gt_root, file_path))
    assert len(gt_files) > 0, 'No annotations found in {}.'.format(gt_root)
    if len(input_files) != len(gt_files):
        logger.warn('Directory {} and {} has {} and {} files, respectively.'
            .format(image_root, gt_root, len(input_files), len(gt_files)))
        input_basenames = [os.path.basename(f)[:-len(image_ext)] for f in
            input_files]
        gt_basenames = [os.path.basename(f)[:-len(gt_ext)] for f in gt_files]
        intersect = list(set(input_basenames) & set(gt_basenames))
        intersect = sorted(intersect)
        logger.warn('Will use their intersection of {} files.'.format(len(
            intersect)))
        input_files = [os.path.join(image_root, f + image_ext) for f in
            intersect]
        gt_files = [os.path.join(gt_root, f + gt_ext) for f in intersect]
    logger.info('Loaded {} images with semantic segmentation from {}'.
        format(len(input_files), image_root))
    dataset_dicts = []
    for img_path, gt_path in zip(input_files, gt_files):
        record = {}
        record['file_name'] = img_path
        record['sem_seg_file_name'] = gt_path
        record['task'] = 'detection'
        record['has_stuff'] = True
        record['dataset_name'] = dataset_name
        dataset_dicts.append(record)
    return dataset_dicts
