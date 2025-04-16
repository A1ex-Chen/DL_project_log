def __len__(self) ->int:
    assert self.check_observation_lengths(
        ), 'Observation lengths are inconsistent'
    return len(list(self._observations.values())[0])
