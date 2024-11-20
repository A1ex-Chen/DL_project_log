def note_encoding_state_to_events(state: NoteEncodingState) ->Sequence[Event]:
    """Output program and pitch events for active notes plus a final tie event."""
    events = []
    for pitch, program in sorted(state.active_pitches.keys(), key=lambda k:
        k[::-1]):
        if state.active_pitches[pitch, program]:
            events += [Event('program', program), Event('pitch', pitch)]
    events.append(Event('tie', 0))
    return events
