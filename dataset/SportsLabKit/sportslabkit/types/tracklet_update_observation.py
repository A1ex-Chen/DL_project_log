def update_observation(self, name: str, value: Any, global_step: (int |
    None)=None) ->None:
    if name in self._observations:
        self._observations[name].append(value)
    else:
        raise ValueError(f"Observation type '{name}' not registered")
