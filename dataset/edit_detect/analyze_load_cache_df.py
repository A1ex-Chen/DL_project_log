def load_cache_df(file: str, err_if_not_exist: bool=False):
    is_file_exist: bool = os.path.isfile(file)
    if err_if_not_exist and not is_file_exist:
        raise ValueError(f'File, {file} does not exist.')
    elif is_file_exist:
        return pd.read_hdf(file, key='df')
    else:
        return None
