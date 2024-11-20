def __init__(self, width=20, with_ptg=True, step_time_average=50,
    speed_unit=Unit.Iter):
    self._width = width
    self._with_ptg = with_ptg
    self._step_time_average = step_time_average
    self._step_times = []
    self._start_time = 0.0
    self._total_size = None
    self._speed_unit = speed_unit
