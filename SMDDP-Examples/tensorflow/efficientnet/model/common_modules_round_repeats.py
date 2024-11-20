def round_repeats(repeats: int, depth_coefficient: float) ->int:
    """Round number of repeats based on depth coefficient."""
    return int(math.ceil(depth_coefficient * repeats))
