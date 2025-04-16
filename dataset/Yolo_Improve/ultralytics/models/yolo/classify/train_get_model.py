def get_model(self, cfg=None, weights=None, verbose=True):
    """Returns a modified PyTorch model configured for training YOLO."""
    model = ClassificationModel(cfg, nc=self.data['nc'], verbose=verbose and
        RANK == -1)
    if weights:
        model.load(weights)
    for m in model.modules():
        if not self.args.pretrained and hasattr(m, 'reset_parameters'):
            m.reset_parameters()
        if isinstance(m, torch.nn.Dropout) and self.args.dropout:
            m.p = self.args.dropout
    for p in model.parameters():
        p.requires_grad = True
    return model
