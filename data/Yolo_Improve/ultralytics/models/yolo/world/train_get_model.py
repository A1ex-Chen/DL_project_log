def get_model(self, cfg=None, weights=None, verbose=True):
    """Return WorldModel initialized with specified config and weights."""
    model = WorldModel(cfg['yaml_file'] if isinstance(cfg, dict) else cfg,
        ch=3, nc=min(self.data['nc'], 80), verbose=verbose and RANK == -1)
    if weights:
        model.load(weights)
    self.add_callback('on_pretrain_routine_end', on_pretrain_routine_end)
    return model
