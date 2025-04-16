def programs_to_midi_classes(tokens, codec):
    """Modifies program events to be the first program in the MIDI class."""
    min_program_id, max_program_id = codec.event_type_range('program')
    is_program = (tokens >= min_program_id) & (tokens <= max_program_id)
    return np.where(is_program, min_program_id + 8 * ((tokens -
        min_program_id) // 8), tokens)
