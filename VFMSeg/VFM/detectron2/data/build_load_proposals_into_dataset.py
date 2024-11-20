def load_proposals_into_dataset(dataset_dicts, proposal_file):
    """
    Load precomputed object proposals into the dataset.

    The proposal file should be a pickled dict with the following keys:

    - "ids": list[int] or list[str], the image ids
    - "boxes": list[np.ndarray], each is an Nx4 array of boxes corresponding to the image id
    - "objectness_logits": list[np.ndarray], each is an N sized array of objectness scores
      corresponding to the boxes.
    - "bbox_mode": the BoxMode of the boxes array. Defaults to ``BoxMode.XYXY_ABS``.

    Args:
        dataset_dicts (list[dict]): annotations in Detectron2 Dataset format.
        proposal_file (str): file path of pre-computed proposals, in pkl format.

    Returns:
        list[dict]: the same format as dataset_dicts, but added proposal field.
    """
    logger = logging.getLogger(__name__)
    logger.info('Loading proposals from: {}'.format(proposal_file))
    with PathManager.open(proposal_file, 'rb') as f:
        proposals = pickle.load(f, encoding='latin1')
    rename_keys = {'indexes': 'ids', 'scores': 'objectness_logits'}
    for key in rename_keys:
        if key in proposals:
            proposals[rename_keys[key]] = proposals.pop(key)
    img_ids = set({str(record['image_id']) for record in dataset_dicts})
    id_to_index = {str(id): i for i, id in enumerate(proposals['ids']) if 
        str(id) in img_ids}
    bbox_mode = BoxMode(proposals['bbox_mode']
        ) if 'bbox_mode' in proposals else BoxMode.XYXY_ABS
    for record in dataset_dicts:
        i = id_to_index[str(record['image_id'])]
        boxes = proposals['boxes'][i]
        objectness_logits = proposals['objectness_logits'][i]
        inds = objectness_logits.argsort()[::-1]
        record['proposal_boxes'] = boxes[inds]
        record['proposal_objectness_logits'] = objectness_logits[inds]
        record['proposal_bbox_mode'] = bbox_mode
    return dataset_dicts
