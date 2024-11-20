def load_numpy(arry: Union[str, np.ndarray], local_path: Optional[str]=None
    ) ->np.ndarray:
    if isinstance(arry, str):
        if local_path is not None:
            return os.path.join(local_path, '/'.join([arry.split('/')[-5],
                arry.split('/')[-2], arry.split('/')[-1]]))
        elif arry.startswith('http://') or arry.startswith('https://'):
            response = requests.get(arry)
            response.raise_for_status()
            arry = np.load(BytesIO(response.content))
        elif os.path.isfile(arry):
            arry = np.load(arry)
        else:
            raise ValueError(
                f'Incorrect path or url, URLs must start with `http://` or `https://`, and {arry} is not a valid path'
                )
    elif isinstance(arry, np.ndarray):
        pass
    else:
        raise ValueError(
            'Incorrect format used for numpy ndarray. Should be an url linking to an image, a local path, or a ndarray.'
            )
    return arry
