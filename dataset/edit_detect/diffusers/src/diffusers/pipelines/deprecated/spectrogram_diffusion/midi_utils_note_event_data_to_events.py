def note_event_data_to_events(state: Optional[NoteEncodingState], value:
    NoteEventData, codec: Codec) ->Sequence[Event]:
    """Convert note event data to a sequence of events."""
    if value.velocity is None:
        return [Event('pitch', value.pitch)]
    else:
        num_velocity_bins = num_velocity_bins_from_codec(codec)
        velocity_bin = velocity_to_bin(value.velocity, num_velocity_bins)
        if value.program is None:
            if state is not None:
                state.active_pitches[value.pitch, 0] = velocity_bin
            return [Event('velocity', velocity_bin), Event('pitch', value.
                pitch)]
        elif value.is_drum:
            return [Event('velocity', velocity_bin), Event('drum', value.pitch)
                ]
        else:
            if state is not None:
                state.active_pitches[value.pitch, value.program] = velocity_bin
            return [Event('program', value.program), Event('velocity',
                velocity_bin), Event('pitch', value.pitch)]
