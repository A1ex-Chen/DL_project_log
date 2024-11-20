@property
def num_classes(self) ->int:
    return sum(er.max_value - er.min_value + 1 for er in self._event_ranges)
