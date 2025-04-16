def check_observation_lengths(self) ->None:
    """check if all value lengths are the same"""
    observation_value_lengths = {k: len(v) for k, v in self._observations.
        items()}
    valid = len(set(observation_value_lengths.values())) == 1
    if not valid:
        logger.warning(
            f'Tracker {self.id} has inconsistent observation lengths:')
        for key, val_len in observation_value_lengths.items():
            logger.warning(f'{key}: {val_len}')
    return valid
