def __init__(self, num_classes: int, backbone: Backbone, anchor_ratios:
    List[Tuple[int, int]], anchor_sizes: List[int], train_rpn_pre_nms_top_n:
    int, train_rpn_post_nms_top_n: int, eval_rpn_pre_nms_top_n: int,
    eval_rpn_post_nms_top_n: int, num_anchor_samples_per_batch: int,
    num_proposal_samples_per_batch: int, num_detections_per_image: int,
    anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
    proposal_nms_threshold: float, detection_nms_threshold: float):
    super().__init__()
    self.num_classes = num_classes
    self.backbone = backbone
    self.anchor_ratios = anchor_ratios
    self.anchor_sizes = anchor_sizes
    self.train_rpn_pre_nms_top_n = train_rpn_pre_nms_top_n
    self.train_rpn_post_nms_top_n = train_rpn_post_nms_top_n
    self.eval_rpn_pre_nms_top_n = eval_rpn_pre_nms_top_n
    self.eval_rpn_post_nms_top_n = eval_rpn_post_nms_top_n
    self.num_anchor_samples_per_batch = num_anchor_samples_per_batch
    self.num_proposal_samples_per_batch = num_proposal_samples_per_batch
    self.num_detections_per_image = num_detections_per_image
    self.anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta
    self.proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
    self.proposal_nms_threshold = proposal_nms_threshold
    self.detection_nms_threshold = detection_nms_threshold
