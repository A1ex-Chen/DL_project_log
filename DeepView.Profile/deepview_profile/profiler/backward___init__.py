def __init__(self, backward_runnable, ag_dict):
    self.run_backward = backward_runnable
    self._ag_dict = ag_dict
