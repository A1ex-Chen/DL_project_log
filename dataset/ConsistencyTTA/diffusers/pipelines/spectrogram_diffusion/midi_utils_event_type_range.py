def event_type_range(self, event_type: str) ->Tuple[int, int]:
    """Return [min_id, max_id] for an event type."""
    offset = 0
    for er in self._event_ranges:
        if event_type == er.type:
            return offset, offset + (er.max_value - er.min_value)
        offset += er.max_value - er.min_value + 1
    raise ValueError(f'Unknown event type: {event_type}')
