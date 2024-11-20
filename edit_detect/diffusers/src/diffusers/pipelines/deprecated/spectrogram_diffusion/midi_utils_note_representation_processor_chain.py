def note_representation_processor_chain(features, codec: Codec,
    note_representation_config: NoteRepresentationConfig):
    tie_token = codec.encode_event(Event('tie', 0))
    state_events_end_token = (tie_token if note_representation_config.
        include_ties else None)
    features = extract_sequence_with_indices(features,
        state_events_end_token=state_events_end_token, feature_key='inputs')
    features = map_midi_programs(features, codec)
    features = run_length_encode_shifts_fn(features, codec,
        state_change_event_types=['velocity', 'program'])
    return features
