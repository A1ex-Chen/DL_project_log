def decode_event_index(self, index: int) ->Event:
    """Decode an event index to an Event."""
    offset = 0
    for er in self._event_ranges:
        if offset <= index <= offset + er.max_value - er.min_value:
            return Event(type=er.type, value=er.min_value + index - offset)
        offset += er.max_value - er.min_value + 1
    raise ValueError(f'Unknown event index: {index}')
