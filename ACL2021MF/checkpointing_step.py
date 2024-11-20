def step(self, metric: Union[float, torch.Tensor]):
    """Serialize checkpoint and update best checkpoint based on metric and mode."""
    if not self._best_metric:
        self._best_metric = metric
    models_state_dict: Dict[str, Any] = {}
    for key in self._models:
        if isinstance(self._models[key], nn.DataParallel):
            models_state_dict[key] = self._models[key].module.state_dict()
        else:
            models_state_dict[key] = self._models[key].state_dict()
    if (self._mode == 'min' and metric <= self._best_metric or self._mode ==
        'max' and metric >= self._best_metric):
        self._best_metric = metric
        torch.save(models_state_dict, os.path.join(self._serialization_dir,
            f'{self._filename_prefix}-best.pth'))
        return True
    return False
