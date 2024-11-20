def make_dir(path: Union[str, os.PathLike], exist_ok: bool=True) ->Union[
    str, os.PathLike]:
    os.makedirs(path, exist_ok=exist_ok)
    return path
