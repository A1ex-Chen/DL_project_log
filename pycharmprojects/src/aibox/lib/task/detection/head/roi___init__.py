def __init__(self, extractor: nn.Module, num_extractor_out: int,
    num_classes: int, num_proposal_samples_per_batch: int,
    num_detections_per_image: int, proposal_smooth_l1_loss_beta: float,
    detection_nms_threshold: float):
    super().__init__()
    self._extractor = extractor
    self._num_classes = num_classes
    self.proposal_class = nn.Linear(num_extractor_out, num_classes)
    self.proposal_transformer = nn.Linear(num_extractor_out, num_classes * 4)
    self._num_proposal_samples_per_batch = num_proposal_samples_per_batch
    self._num_detections_per_image = num_detections_per_image
    self._proposal_smooth_l1_loss_beta = proposal_smooth_l1_loss_beta
    self._detection_nms_threshold = detection_nms_threshold
    self._transformer_normalize_mean = torch.tensor([0.0, 0.0, 0.0, 0.0],
        dtype=torch.float)
    self._transformer_normalize_std = torch.tensor([0.1, 0.1, 0.2, 0.2],
        dtype=torch.float)
