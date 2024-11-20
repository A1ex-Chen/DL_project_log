def build_tracker_head(cfg: CfgNode_) ->BaseTracker:
    """
    Build a tracker head from `cfg.TRACKER_HEADS.TRACKER_NAME`.

    Args:
        cfg: D2 CfgNode, config file with tracker information
    Return:
        tracker object
    """
    name = cfg.TRACKER_HEADS.TRACKER_NAME
    tracker_class = TRACKER_HEADS_REGISTRY.get(name)
    return tracker_class(cfg)
