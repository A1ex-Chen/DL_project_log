def is_valid_file(x: str) ->bool:
    return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
