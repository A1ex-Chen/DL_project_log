def update_noisy_image(self, values: torch.Tensor):
    if SafetensorRecorder.NOISY_IMAGE_KEY in self.__data__:
        self.__data__[SafetensorRecorder.NOISY_IMAGE_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.NOISY_IMAGE_KEY] = [values]
