@configurable
def __init__(self, *, backbone: Backbone, sem_seg_head: nn.Module,
    criterion: nn.Module, losses: dict, num_queries: int,
    object_mask_threshold: float, overlap_threshold: float, metadata,
    task_switch: dict, phrase_prob: float, size_divisibility: int,
    sem_seg_postprocess_before_inference: bool, pixel_mean: Tuple[float],
    pixel_std: Tuple[float], semantic_on: bool, panoptic_on: bool,
    instance_on: bool, test_topk_per_image: int, train_dataset_name: str,
    interactive_mode: str, interactive_iter: str, dilation_kernel: torch.Tensor
    ):
    super().__init__()
    self.backbone = backbone
    self.sem_seg_head = sem_seg_head
    self.criterion = criterion
    self.losses = losses
    self.num_queries = num_queries
    self.overlap_threshold = overlap_threshold
    self.object_mask_threshold = object_mask_threshold
    self.metadata = metadata
    if size_divisibility < 0:
        size_divisibility = self.backbone.size_divisibility
    self.size_divisibility = size_divisibility
    self.sem_seg_postprocess_before_inference = (
        sem_seg_postprocess_before_inference)
    self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(-1, 1,
        1), False)
    self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(-1, 1, 1
        ), False)
    self.semantic_on = semantic_on
    self.instance_on = instance_on
    self.panoptic_on = panoptic_on
    self.task_switch = task_switch
    self.phrase_prob = phrase_prob
    self.test_topk_per_image = test_topk_per_image
    self.train_class_names = None
    self.interactive_mode = interactive_mode
    self.interactive_iter = interactive_iter
    if not self.semantic_on:
        assert self.sem_seg_postprocess_before_inference
    self.register_buffer('dilation_kernel', dilation_kernel)
