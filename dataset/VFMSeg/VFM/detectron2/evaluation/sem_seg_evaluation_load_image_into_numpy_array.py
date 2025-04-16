def load_image_into_numpy_array(filename: str, copy: bool=False, dtype:
    Optional[Union[np.dtype, str]]=None) ->np.ndarray:
    with PathManager.open(filename, 'rb') as f:
        array = np.array(Image.open(f), copy=copy, dtype=dtype)
    return array
