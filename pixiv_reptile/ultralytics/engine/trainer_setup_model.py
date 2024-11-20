def setup_model(self):
    """Load/create/download model for any task."""
    if isinstance(self.model, torch.nn.Module):
        return
    cfg, weights = self.model, None
    ckpt = None
    if str(self.model).endswith('.pt'):
        weights, ckpt = attempt_load_one_weight(self.model)
        cfg = weights.yaml
    elif isinstance(self.args.pretrained, (str, Path)):
        weights, _ = attempt_load_one_weight(self.args.pretrained)
    self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)
    return ckpt
