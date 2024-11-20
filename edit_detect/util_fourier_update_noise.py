def update_noise(self, values: torch.Tensor):
    if SafetensorRecorder.NOISE_KEY in self.__data__:
        self.__data__[SafetensorRecorder.NOISE_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.NOISE_KEY] = [values]
