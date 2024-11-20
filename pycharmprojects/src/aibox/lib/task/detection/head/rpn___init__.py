def __init__(self, extractor: nn.Module, num_extractor_out: int,
    anchor_ratios: List[Tuple[int, int]], anchor_sizes: List[int],
    train_pre_nms_top_n: int, train_post_nms_top_n: int, eval_pre_nms_top_n:
    int, eval_post_nms_top_n: int, num_anchor_samples_per_batch: int,
    anchor_smooth_l1_loss_beta: float, proposal_nms_threshold: float):
    super().__init__()
    self._extractor = extractor
    self._anchor_ratios = anchor_ratios
    self._anchor_sizes = anchor_sizes
    num_anchor_ratios = len(self._anchor_ratios)
    num_anchor_sizes = len(self._anchor_sizes)
    num_anchors = num_anchor_ratios * num_anchor_sizes
    self._train_pre_nms_top_n = train_pre_nms_top_n
    self._train_post_nms_top_n = train_post_nms_top_n
    self._eval_pre_nms_top_n = eval_pre_nms_top_n
    self._eval_post_nms_top_n = eval_post_nms_top_n
    self._num_anchor_samples_per_batch = num_anchor_samples_per_batch
    self._anchor_smooth_l1_loss_beta = anchor_smooth_l1_loss_beta
    self._proposal_nms_threshold = proposal_nms_threshold
    self.anchor_objectness = nn.Conv2d(in_channels=num_extractor_out,
        out_channels=num_anchors * 2, kernel_size=1)
    self.anchor_transformer = nn.Conv2d(in_channels=num_extractor_out,
        out_channels=num_anchors * 4, kernel_size=1)
