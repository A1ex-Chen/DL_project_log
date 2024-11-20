@property
def step_index(self):
    """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
    return self._step_index
