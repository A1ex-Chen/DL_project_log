def print_bar(self, finished_size=1, pre_string=None, post_string=None):
    self._step_times.append(time.time() - self._current_time)
    self._finished_sizes.append(finished_size)
    self._time_elapsed += self._step_times[-1]
    start_time_str = second_to_time_str(self._time_elapsed)
    time_per_size = np.array(self._step_times[-self._step_time_average:])
    time_per_size /= np.array(self._finished_sizes[-self._step_time_average:])
    average_step_time = np.mean(time_per_size) + 1e-06
    if self._speed_unit == Unit.Iter:
        speed_str = '{:.2f}it/s'.format(1 / average_step_time)
    elif self._speed_unit == Unit.Byte:
        size, size_unit = convert_size(1 / average_step_time)
        speed_str = '{:.2f}{}/s'.format(size, size_unit)
    else:
        raise ValueError('unknown speed unit')
    remain_time = (self._total_size - self._progress) * average_step_time
    remain_time_str = second_to_time_str(remain_time)
    time_str = start_time_str + '>' + remain_time_str
    prog_str = progress_str((self._progress + 1) / self._total_size,
        speed_str, time_str, width=self._width, with_ptg=self._with_ptg)
    self._progress += finished_size
    if pre_string is not None:
        prog_str = pre_string + prog_str
    if post_string is not None:
        prog_str += post_string
    if self._progress >= self._total_size:
        print(prog_str + '   ')
    else:
        print(prog_str + '   ', end='\r')
    self._current_time = time.time()
