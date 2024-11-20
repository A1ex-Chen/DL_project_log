def end(self, model_name: str, sub_calls: Dict[str, int]) ->None:
    end_time = time.perf_counter()
    if model_name not in self.recording_hook:
        raise ValueError(
            f'Attempting to end {model_name} which was never started.')
    start_time = self.recording_hook.pop(model_name)
    duration = end_time - start_time
    self.durations[model_name].append(duration)
    net_duration = self._calculate_net_cost(duration, sub_calls)
    self.net_durations[model_name].append(net_duration)
