def __init__(self, model, data_loader, optimizer, grad_scaler=None):
    """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
    unsupported = (
        'AMPTrainer does not support single-process multi-device training!')
    if isinstance(model, DistributedDataParallel):
        assert not (model.device_ids and len(model.device_ids) > 1
            ), unsupported
    assert not isinstance(model, DataParallel), unsupported
    super().__init__(model, data_loader, optimizer)
    if grad_scaler is None:
        from torch.cuda.amp import GradScaler
        grad_scaler = GradScaler()
    self.grad_scaler = grad_scaler
