@configurable
def __init__(self, *, in_features: List[str], head: nn.Module,
    anchor_generator: nn.Module, anchor_matcher: Matcher, box2box_transform:
    Box2BoxTransform, batch_size_per_image: int, positive_fraction: float,
    pre_nms_topk: Tuple[float, float], post_nms_topk: Tuple[float, float],
    nms_thresh: float=0.7, min_box_size: float=0.0, anchor_boundary_thresh:
    float=-1.0, loss_weight: Union[float, Dict[str, float]]=1.0,
    box_reg_loss_type: str='smooth_l1', smooth_l1_beta: float=0.0):
    """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            batch_size_per_image (int): number of anchors per image to sample for training
            positive_fraction (float): fraction of foreground anchors to sample for training
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all rpn losses together, or a dict of individual weightings. Valid dict keys are:
                    "loss_rpn_cls" - applied to classification loss
                    "loss_rpn_loc" - applied to box regression loss
            box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou".
            smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
                use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
        """
    super().__init__()
    self.in_features = in_features
    self.rpn_head = head
    self.anchor_generator = anchor_generator
    self.anchor_matcher = anchor_matcher
    self.box2box_transform = box2box_transform
    self.batch_size_per_image = batch_size_per_image
    self.positive_fraction = positive_fraction
    self.pre_nms_topk = {(True): pre_nms_topk[0], (False): pre_nms_topk[1]}
    self.post_nms_topk = {(True): post_nms_topk[0], (False): post_nms_topk[1]}
    self.nms_thresh = nms_thresh
    self.min_box_size = float(min_box_size)
    self.anchor_boundary_thresh = anchor_boundary_thresh
    if isinstance(loss_weight, float):
        loss_weight = {'loss_rpn_cls': loss_weight, 'loss_rpn_loc': loss_weight
            }
    self.loss_weight = loss_weight
    self.box_reg_loss_type = box_reg_loss_type
    self.smooth_l1_beta = smooth_l1_beta
