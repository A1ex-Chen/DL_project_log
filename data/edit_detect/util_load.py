def load(self, path: Union[str, os.PathLike], enable_update: bool=False
    ) ->'SafetensorRecorder':
    loaded_data: dict = {}
    with safe_open(path, framework='pt', device='cpu') as f:
        for k in f.keys():
            if enable_update:
                loaded_data[k] = [f.get_tensor(k)]
            else:
                loaded_data[k] = f.get_tensor(k)
        self.__unpack_internal__(input=loaded_data)
    return self
