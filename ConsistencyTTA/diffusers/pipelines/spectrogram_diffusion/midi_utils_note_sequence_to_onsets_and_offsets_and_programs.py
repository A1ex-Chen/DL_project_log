def note_sequence_to_onsets_and_offsets_and_programs(ns: note_seq.NoteSequence
    ) ->Tuple[Sequence[float], Sequence[NoteEventData]]:
    """Extract onset & offset times and pitches & programs from a NoteSequence.

    The onset & offset times will not necessarily be in sorted order.

    Args:
      ns: NoteSequence from which to extract onsets and offsets.

    Returns:
      times: A list of note onset and offset times. values: A list of NoteEventData objects where velocity is zero for
      note
          offsets.
    """
    notes = sorted(ns.notes, key=lambda note: (note.is_drum, note.program,
        note.pitch))
    times = [note.end_time for note in notes if not note.is_drum] + [note.
        start_time for note in notes]
    values = [NoteEventData(pitch=note.pitch, velocity=0, program=note.
        program, is_drum=False) for note in notes if not note.is_drum] + [
        NoteEventData(pitch=note.pitch, velocity=note.velocity, program=
        note.program, is_drum=note.is_drum) for note in notes]
    return times, values
