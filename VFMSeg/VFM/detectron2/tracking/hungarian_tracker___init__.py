@configurable
def __init__(self, video_height: int, video_width: int, max_num_instances:
    int=200, max_lost_frame_count: int=0, min_box_rel_dim: float=0.02,
    min_instance_period: int=1, **kwargs):
    """
        Args:
        video_height: height the video frame
        video_width: width of the video frame
        max_num_instances: maximum number of id allowed to be tracked
        max_lost_frame_count: maximum number of frame an id can lost tracking
                              exceed this number, an id is considered as lost
                              forever
        min_box_rel_dim: a percentage, smaller than this dimension, a bbox is
                         removed from tracking
        min_instance_period: an instance will be shown after this number of period
                             since its first showing up in the video
        """
    super().__init__(**kwargs)
    self._video_height = video_height
    self._video_width = video_width
    self._max_num_instances = max_num_instances
    self._max_lost_frame_count = max_lost_frame_count
    self._min_box_rel_dim = min_box_rel_dim
    self._min_instance_period = min_instance_period
