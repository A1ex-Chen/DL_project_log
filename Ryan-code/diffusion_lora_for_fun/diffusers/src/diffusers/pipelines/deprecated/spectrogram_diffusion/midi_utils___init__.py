def __init__(self):
    self.codec = Codec(max_shift_steps=DEFAULT_MAX_SHIFT_SECONDS *
        DEFAULT_STEPS_PER_SECOND, steps_per_second=DEFAULT_STEPS_PER_SECOND,
        event_ranges=[EventRange('pitch', note_seq.MIN_MIDI_PITCH, note_seq
        .MAX_MIDI_PITCH), EventRange('velocity', 0,
        DEFAULT_NUM_VELOCITY_BINS), EventRange('tie', 0, 0), EventRange(
        'program', note_seq.MIN_MIDI_PROGRAM, note_seq.MAX_MIDI_PROGRAM),
        EventRange('drum', note_seq.MIN_MIDI_PITCH, note_seq.MAX_MIDI_PITCH)])
    self.tokenizer = Tokenizer(self.codec.num_classes)
    self.note_representation_config = NoteRepresentationConfig(onsets_only=
        False, include_ties=True)
