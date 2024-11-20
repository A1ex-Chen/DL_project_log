def encode_event(self, event: Event) ->int:
    """Encode an event to an index."""
    offset = 0
    for er in self._event_ranges:
        if event.type == er.type:
            if not er.min_value <= event.value <= er.max_value:
                raise ValueError(
                    f'Event value {event.value} is not within valid range [{er.min_value}, {er.max_value}] for type {event.type}'
                    )
            return offset + event.value - er.min_value
        offset += er.max_value - er.min_value + 1
    raise ValueError(f'Unknown event type: {event.type}')
