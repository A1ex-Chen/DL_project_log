@register_model
def get_segmentation_model(cfg, **kwargs):
    return SEEM_Model(cfg)
