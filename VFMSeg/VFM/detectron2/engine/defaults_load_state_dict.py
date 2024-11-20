def load_state_dict(self, state_dict):
    super().load_state_dict(state_dict)
    self._trainer.load_state_dict(state_dict['_trainer'])
