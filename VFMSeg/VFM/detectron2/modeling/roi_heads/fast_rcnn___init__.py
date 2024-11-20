@configurable
def __init__(self, input_shape: ShapeSpec, *, box2box_transform,
    num_classes: int, test_score_thresh: float=0.0, test_nms_thresh: float=
    0.5, test_topk_per_image: int=100, cls_agnostic_bbox_reg: bool=False,
    smooth_l1_beta: float=0.0, box_reg_loss_type: str='smooth_l1',
    loss_weight: Union[float, Dict[str, float]]=1.0):
    """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou",
                "diou", "ciou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
    super().__init__()
    if isinstance(input_shape, int):
        input_shape = ShapeSpec(channels=input_shape)
    self.num_classes = num_classes
    input_size = input_shape.channels * (input_shape.width or 1) * (input_shape
        .height or 1)
    self.cls_score = nn.Linear(input_size, num_classes + 1)
    num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
    box_dim = len(box2box_transform.weights)
    self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)
    nn.init.normal_(self.cls_score.weight, std=0.01)
    nn.init.normal_(self.bbox_pred.weight, std=0.001)
    for l in [self.cls_score, self.bbox_pred]:
        nn.init.constant_(l.bias, 0)
    self.box2box_transform = box2box_transform
    self.smooth_l1_beta = smooth_l1_beta
    self.test_score_thresh = test_score_thresh
    self.test_nms_thresh = test_nms_thresh
    self.test_topk_per_image = test_topk_per_image
    self.box_reg_loss_type = box_reg_loss_type
    if isinstance(loss_weight, float):
        loss_weight = {'loss_cls': loss_weight, 'loss_box_reg': loss_weight}
    self.loss_weight = loss_weight
