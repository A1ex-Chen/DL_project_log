def is_mot(filename: PathLike) ->bool:
    """Return True if the file is MOT format.

    Args:
        filename(PathLike): Path to file.

    Returns:
        is_mot(bool): True if the file is MOT format.
    """
    try:
        with open(filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            first_line = next(reader)
        return ['frame', 'id', 'bb_left', 'bb_top', 'bb_width', 'bb_height'
            ] == first_line
    except Exception:
        return False
