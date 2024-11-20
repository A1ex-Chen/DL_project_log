def _create_state_dict(self):
    return {STATE_DICT_KEY: self.model.module.state_dict() if self.
        is_parallel else self.model.state_dict(), OPTIMIZER_STATE_DICT_KEY:
        self.optimizer.state_dict()}
