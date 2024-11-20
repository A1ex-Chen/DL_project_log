def predict(self, observations: dict[str, [float | ndarray]], states: dict[
    str, float | ndarray]=None) ->tuple[ndarray, dict[str, float | ndarray]]:
    boxes = np.array(observations.get('box', None))
    scores = np.array(observations.get('score', 1))
    new_states = states.copy() if states else {}
    if new_states.get('x') is None:
        new_states.update(self.get_initial_kalman_filter_states(boxes[-1]))
        for box, score in zip(boxes, scores):
            new_states['x'], new_states['P'] = predict(x=new_states['x'], P
                =new_states['P'], F=new_states['F'], Q=new_states['Q'])
            new_states['R'] = self._initialize_measurement_noise_covariance(
                score)
            new_states['x'], new_states['P'] = update(new_states['x'],
                new_states['P'], box, new_states['R'], new_states['H'])
    else:
        new_states['x'], new_states['P'] = predict(states['x'], states['P'],
            states['F'], states['Q'])
        new_states['R'] = self._initialize_measurement_noise_covariance(scores
            [-1])
        new_states['x'], new_states['P'] = update(new_states['x'],
            new_states['P'], boxes[-1], new_states['R'], states['H'])
    pred = new_states['x'][:4]
    pred = torch.tensor(pred).unsqueeze(0)
    return pred, new_states
