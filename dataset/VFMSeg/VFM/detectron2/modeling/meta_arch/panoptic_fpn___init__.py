@configurable
def __init__(self, *, sem_seg_head: nn.Module, combine_overlap_thresh:
    float=0.5, combine_stuff_area_thresh: float=4096,
    combine_instances_score_thresh: float=0.5, **kwargs):
    """
        NOTE: this interface is experimental.

        Args:
            sem_seg_head: a module for the semantic segmentation head.
            combine_overlap_thresh: combine masks into one instances if
                they have enough overlap
            combine_stuff_area_thresh: ignore stuff areas smaller than this threshold
            combine_instances_score_thresh: ignore instances whose score is
                smaller than this threshold

        Other arguments are the same as :class:`GeneralizedRCNN`.
        """
    super().__init__(**kwargs)
    self.sem_seg_head = sem_seg_head
    self.combine_overlap_thresh = combine_overlap_thresh
    self.combine_stuff_area_thresh = combine_stuff_area_thresh
    self.combine_instances_score_thresh = combine_instances_score_thresh
