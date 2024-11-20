def drop_programs(tokens, codec: Codec):
    """Drops program change events from a token sequence."""
    min_program_id, max_program_id = codec.event_type_range('program')
    return tokens[(tokens < min_program_id) | (tokens > max_program_id)]
