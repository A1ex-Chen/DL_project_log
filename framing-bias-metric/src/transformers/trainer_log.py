def log(self, logs: Dict[str, float]) ->None:
    """
        Log :obj:`logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (:obj:`Dict[str, float]`):
                The values to log.
        """
    if self.state.epoch is not None:
        logs['epoch'] = self.state.epoch
    self.control = self.callback_handler.on_log(self.args, self.state, self
        .control, logs)
    output = {**logs, **{'step': self.state.global_step}}
    self.state.log_history.append(output)
