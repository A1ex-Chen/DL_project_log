def __getattr__(self, name: str) ->Any:
    if name in self._observations:
        return self.get_observation(name)
    elif name in self._states:
        return self.get_state(name)
    else:
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'")
