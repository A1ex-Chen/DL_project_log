def process_before_saving(self, mode: str):
    if mode == SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK:
        for key, val in self.__data__.items():
            self.__data__[key] = torch.stack(val, dim=0)
    elif mode == SafetensorRecorder.PROC_BEF_SAVE_MODE_CAT:
        for key, val in self.__data__.items():
            self.__data__[key] = torch.cat(val, dim=0)
    else:
        raise ValueError(
            f'Arguement mode should be {SafetensorRecorder.PROC_BEF_SAVE_MODE_STACK} or {SafetensorRecorder.PROC_BEF_SAVE_MODE_CAT}'
            )
