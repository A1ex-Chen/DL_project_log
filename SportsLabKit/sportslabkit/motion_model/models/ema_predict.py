def predict(self, observations: dict[str, Any], states: dict[str, float |
    np.ndarray[Any, Any]]=...) ->tuple[np.ndarray[Any, Any], dict[str, 
    float | np.ndarray[Any, Any]]]:
    gamma = self.gamma
    boxes = np.array(observations.get('box', None))
    EMA_t = states['EMA_t']
    if EMA_t is None:
        for box in boxes:
            EMA_t = box if EMA_t is None else gamma * EMA_t + (1 - gamma) * box
    else:
        box = boxes[-1:].squeeze()
        EMA_t = gamma * EMA_t + (1 - gamma) * box
    new_states = {'EMA_t': EMA_t}
    return EMA_t, new_states
