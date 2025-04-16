def is_shift_event_index(self, index: int) ->bool:
    return (self._shift_range.min_value <= index and index <= self.
        _shift_range.max_value)
