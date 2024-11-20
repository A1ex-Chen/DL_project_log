def save(self, path: Union[str, os.PathLike], file_ext: str='safetensors',
    proc_mode: str=PROC_BEF_SAVE_MODE_CAT) ->None:
    if SafetensorRecorder.RESIDUAL_KEY not in self.__data__:
        path = f'{path}_woRes'
    if SafetensorRecorder.SEQ_KEY not in self.__data__:
        path = f'{path}_woSeq'
    file_path: str = f'{path}.{file_ext}'
    if file_ext is None or file_ext == '':
        file_path: str = path
    self.process_before_saving(mode=proc_mode)
    save_file(self.__pack_internal__(), file_path)
