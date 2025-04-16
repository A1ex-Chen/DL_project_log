def subsegment(self, start_time=None, end_time=None):
    """Cut the AudioSegment between given boundaries.

        Note that this is an in-place transformation.
        :param start_time: Beginning of subsegment in seconds.
        :type start_time: float
        :param end_time: End of subsegment in seconds.
        :type end_time: float
        :raise ValueError: If start_time or end_time is incorrectly set, e.g. out
                                             of bounds in time.
        """
    start_time = 0.0 if start_time is None else start_time
    end_time = self.duration if end_time is None else end_time
    if start_time < 0.0:
        start_time = self.duration + start_time
    if end_time < 0.0:
        end_time = self.duration + end_time
    if start_time < 0.0:
        raise ValueError(
            'The slice start position (%f s) is out of bounds.' % start_time)
    if end_time < 0.0:
        raise ValueError('The slice end position (%f s) is out of bounds.' %
            end_time)
    if start_time > end_time:
        raise ValueError(
            'The slice start position (%f s) is later than the end position (%f s).'
             % (start_time, end_time))
    if end_time > self.duration:
        raise ValueError(
            'The slice end position (%f s) is out of bounds (> %f s)' % (
            end_time, self.duration))
    start_sample = int(round(start_time * self._sample_rate))
    end_sample = int(round(end_time * self._sample_rate))
    self._samples = self._samples[start_sample:end_sample]
