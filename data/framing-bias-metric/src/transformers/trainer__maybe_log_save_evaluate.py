def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
    if self.control.should_log:
        logs: Dict[str, float] = {}
        tr_loss_scalar = tr_loss.item()
        tr_loss -= tr_loss
        logs['loss'] = tr_loss_scalar / (self.state.global_step - self.
            _globalstep_last_logged)
        logs['learning_rate'] = self.lr_scheduler.get_last_lr()[0
            ] if version.parse(torch.__version__) >= version.parse('1.4'
            ) else self.lr_scheduler.get_lr()[0]
        self._total_loss_scalar += tr_loss_scalar
        self._globalstep_last_logged = self.state.global_step
        self.log(logs)
    metrics = None
    if self.control.should_evaluate:
        metrics = self.evaluate()
        self._report_to_hp_search(trial, epoch, metrics)
    if self.control.should_save:
        self._save_checkpoint(model, trial, metrics=metrics)
        self.control = self.callback_handler.on_save(self.args, self.state,
            self.control)
