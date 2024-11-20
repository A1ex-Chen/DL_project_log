def update_ts(self, values: torch.Tensor):
    if SafetensorRecorder.TS_KEY in self.__data__:
        self.__data__[SafetensorRecorder.TS_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.TS_KEY] = [values]
