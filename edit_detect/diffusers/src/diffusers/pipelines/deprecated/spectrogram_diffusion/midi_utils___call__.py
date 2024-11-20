def __call__(self, midi: Union[bytes, os.PathLike, str]):
    if not isinstance(midi, bytes):
        with open(midi, 'rb') as f:
            midi = f.read()
    ns = note_seq.midi_to_note_sequence(midi)
    ns_sus = note_seq.apply_sustain_control_changes(ns)
    for note in ns_sus.notes:
        if not note.is_drum:
            note.program = program_to_slakh_program(note.program)
    samples = np.zeros(int(ns_sus.total_time * SAMPLE_RATE))
    _, frame_times = audio_to_frames(samples, HOP_SIZE, FRAME_RATE)
    times, values = note_sequence_to_onsets_and_offsets_and_programs(ns_sus)
    events = encode_and_index_events(state=NoteEncodingState(), event_times
        =times, event_values=values, frame_times=frame_times, codec=self.
        codec, encode_event_fn=note_event_data_to_events,
        encoding_state_to_events_fn=note_encoding_state_to_events)
    events = [note_representation_processor_chain(event, self.codec, self.
        note_representation_config) for event in events]
    input_tokens = [self.tokenizer.encode(event['inputs']) for event in events]
    return input_tokens
