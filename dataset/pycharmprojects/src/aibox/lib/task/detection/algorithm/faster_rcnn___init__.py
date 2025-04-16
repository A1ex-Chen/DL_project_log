def __init__(self, num_classes: int, backbone: Backbone, anchor_ratios:
    List[Tuple[int, int]], anchor_sizes: List[int], train_rpn_pre_nms_top_n:
    int, train_rpn_post_nms_top_n: int, eval_rpn_pre_nms_top_n: int,
    eval_rpn_post_nms_top_n: int, num_anchor_samples_per_batch: int,
    num_proposal_samples_per_batch: int, num_detections_per_image: int,
    anchor_smooth_l1_loss_beta: float, proposal_smooth_l1_loss_beta: float,
    proposal_nms_threshold: float, detection_nms_threshold: float):
    super().__init__(num_classes, backbone, anchor_ratios, anchor_sizes,
        train_rpn_pre_nms_top_n, train_rpn_post_nms_top_n,
        eval_rpn_pre_nms_top_n, eval_rpn_post_nms_top_n,
        num_anchor_samples_per_batch, num_proposal_samples_per_batch,
        num_detections_per_image, anchor_smooth_l1_loss_beta,
        proposal_smooth_l1_loss_beta, proposal_nms_threshold,
        detection_nms_threshold)
    self.body, num_body_out = self._build_body()
    self.rpn_head, num_rpn_extractor_out = self._build_rpn_head(
        num_extractor_in=num_body_out)
    self.roi_head = self._build_roi_head()
    self._roi_align = RoIAlign(output_size=(14, 14), spatial_scale=1 / 16,
        sampling_ratio=0)
