def build_detector(cfg):
    return DETECTORS[cfg.MODEL.DETECTOR](cfg)
