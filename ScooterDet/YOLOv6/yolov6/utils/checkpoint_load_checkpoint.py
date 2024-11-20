def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    """Load model from checkpoint file."""
    LOGGER.info('Loading checkpoint from {}'.format(weights))
    ckpt = torch.load(weights, map_location=map_location)
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        LOGGER.info('\nFusing model...')
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model
