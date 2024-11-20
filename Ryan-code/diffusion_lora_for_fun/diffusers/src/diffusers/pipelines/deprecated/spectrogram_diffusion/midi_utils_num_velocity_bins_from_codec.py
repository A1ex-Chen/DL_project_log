def num_velocity_bins_from_codec(codec: Codec):
    """Get number of velocity bins from event codec."""
    lo, hi = codec.event_type_range('velocity')
    return hi - lo
