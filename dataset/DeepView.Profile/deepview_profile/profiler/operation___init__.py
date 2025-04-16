def __init__(self, warm_up=3, measure_for=3):
    self._warm_up = warm_up
    self._measure_for = measure_for
    self._start_event = torch.cuda.Event(enable_timing=True)
    self._end_event = torch.cuda.Event(enable_timing=True)
