@configurable
def __init__(self, is_train: bool, *, augmentations: List[Union[T.
    Augmentation, T.Transform]], image_format: str, use_instance_mask: bool
    =False, use_keypoint: bool=False, instance_mask_format: str='polygon',
    keypoint_hflip_indices: Optional[np.ndarray]=None,
    precomputed_proposal_topk: Optional[int]=None, recompute_boxes: bool=False
    ):
    """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
    if recompute_boxes:
        assert use_instance_mask, 'recompute_boxes requires instance masks'
    self.is_train = is_train
    self.augmentations = T.AugmentationList(augmentations)
    self.image_format = image_format
    self.use_instance_mask = use_instance_mask
    self.instance_mask_format = instance_mask_format
    self.use_keypoint = use_keypoint
    self.keypoint_hflip_indices = keypoint_hflip_indices
    self.proposal_topk = precomputed_proposal_topk
    self.recompute_boxes = recompute_boxes
    logger = logging.getLogger(__name__)
    mode = 'training' if is_train else 'inference'
    logger.info(
        f'[DatasetMapper] Augmentations used in {mode}: {augmentations}')
