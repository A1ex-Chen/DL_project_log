def load_model_state(self, state):
    if not state is None:
        self.model.load_state_dict(state)
