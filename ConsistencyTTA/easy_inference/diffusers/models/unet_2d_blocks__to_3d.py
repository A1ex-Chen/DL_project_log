def _to_3d(self, hidden_states, height, weight):
    return hidden_states.permute(0, 2, 3, 1).reshape(hidden_states.shape[0],
        height * weight, -1)
