@configurable
def __init__(self, *, box_in_features: List[str], box_pooler: ROIPooler,
    box_head: nn.Module, box_predictor: nn.Module, mask_in_features:
    Optional[List[str]]=None, mask_pooler: Optional[ROIPooler]=None,
    mask_head: Optional[nn.Module]=None, keypoint_in_features: Optional[
    List[str]]=None, keypoint_pooler: Optional[ROIPooler]=None,
    keypoint_head: Optional[nn.Module]=None, train_on_pred_boxes: bool=
    False, **kwargs):
    """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
    super().__init__(**kwargs)
    self.in_features = self.box_in_features = box_in_features
    self.box_pooler = box_pooler
    self.box_head = box_head
    self.box_predictor = box_predictor
    self.mask_on = mask_in_features is not None
    if self.mask_on:
        self.mask_in_features = mask_in_features
        self.mask_pooler = mask_pooler
        self.mask_head = mask_head
    self.keypoint_on = keypoint_in_features is not None
    if self.keypoint_on:
        self.keypoint_in_features = keypoint_in_features
        self.keypoint_pooler = keypoint_pooler
        self.keypoint_head = keypoint_head
    self.train_on_pred_boxes = train_on_pred_boxes
