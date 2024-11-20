def generate_identified_filename(filename: Path, identifier: str) ->Path:
    """
    Append a string-identifier at the end (before the extension, if any) to the provided filepath

    Args:
        filename: pathlib.Path The actual path object we would like to add an identifier suffix
        identifier: The suffix to add

    Returns: String with concatenated identifier at the end of the filename
    """
    return filename.parent.joinpath(filename.stem + identifier).with_suffix(
        filename.suffix)
