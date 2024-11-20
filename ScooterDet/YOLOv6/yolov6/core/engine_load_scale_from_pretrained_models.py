@staticmethod
def load_scale_from_pretrained_models(cfg, device):
    weights = cfg.model.scales
    scales = None
    if not weights:
        LOGGER.error('ERROR: No scales provided to init RepOptimizer!')
    else:
        ckpt = torch.load(weights, map_location=device)
        scales = extract_scales(ckpt)
    return scales
