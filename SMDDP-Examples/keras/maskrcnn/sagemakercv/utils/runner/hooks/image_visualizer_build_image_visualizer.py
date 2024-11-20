@HOOKS.register('ImageVisualizer')
def build_image_visualizer(cfg):
    return ImageVisualizer(interval=cfg.MODEL.INFERENCE.VISUALIZE_INTERVAL,
        threshold=cfg.MODEL.INFERENCE.VISUALIZE_THRESHOLD)
