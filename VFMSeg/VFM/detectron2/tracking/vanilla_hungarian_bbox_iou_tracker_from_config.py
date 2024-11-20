@classmethod
def from_config(cls, cfg: CfgNode_):
    """
        Old style initialization using CfgNode

        Args:
            cfg: D2 CfgNode, config file
        Return:
            dictionary storing arguments for __init__ method
        """
    assert 'VIDEO_HEIGHT' in cfg.TRACKER_HEADS
    assert 'VIDEO_WIDTH' in cfg.TRACKER_HEADS
    video_height = cfg.TRACKER_HEADS.get('VIDEO_HEIGHT')
    video_width = cfg.TRACKER_HEADS.get('VIDEO_WIDTH')
    max_num_instances = cfg.TRACKER_HEADS.get('MAX_NUM_INSTANCES', 200)
    max_lost_frame_count = cfg.TRACKER_HEADS.get('MAX_LOST_FRAME_COUNT', 0)
    min_box_rel_dim = cfg.TRACKER_HEADS.get('MIN_BOX_REL_DIM', 0.02)
    min_instance_period = cfg.TRACKER_HEADS.get('MIN_INSTANCE_PERIOD', 1)
    track_iou_threshold = cfg.TRACKER_HEADS.get('TRACK_IOU_THRESHOLD', 0.5)
    return {'_target_':
        'detectron2.tracking.vanilla_hungarian_bbox_iou_tracker.VanillaHungarianBBoxIOUTracker'
        , 'video_height': video_height, 'video_width': video_width,
        'max_num_instances': max_num_instances, 'max_lost_frame_count':
        max_lost_frame_count, 'min_box_rel_dim': min_box_rel_dim,
        'min_instance_period': min_instance_period, 'track_iou_threshold':
        track_iou_threshold}
