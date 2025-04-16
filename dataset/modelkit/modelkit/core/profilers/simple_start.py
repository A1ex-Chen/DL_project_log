def start(self, model_name: str) ->None:
    if model_name in self.recording_hook:
        raise ValueError(
            f'Attempting to start {model_name} which has already started.')
    self.recording_hook[model_name] = time.perf_counter()
