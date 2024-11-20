def update_reconst(self, values: torch.Tensor):
    if SafetensorRecorder.RECONST_KEY in self.__data__:
        self.__data__[SafetensorRecorder.RECONST_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.RECONST_KEY] = [values]
