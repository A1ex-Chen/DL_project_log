def before_train(self):
    self._optimizer = self._optimizer or self.trainer.optimizer
    if isinstance(self.scheduler, ParamScheduler):
        self._scheduler = LRMultiplier(self._optimizer, self.scheduler,
            self.trainer.max_iter, last_iter=self.trainer.iter - 1)
    self._best_param_group_id = LRScheduler.get_best_param_group_id(self.
        _optimizer)
