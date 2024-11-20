def update_residual(self, values: torch.Tensor):
    if SafetensorRecorder.RESIDUAL_KEY in self.__data__:
        self.__data__[DirectRecorder.RESIDUAL_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.RESIDUAL_KEY] = [values]
