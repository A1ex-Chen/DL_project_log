def _to_4d(self, hidden_states, height, weight):
    return hidden_states.permute(0, 2, 1).reshape(hidden_states.shape[0], -
        1, height, weight)
