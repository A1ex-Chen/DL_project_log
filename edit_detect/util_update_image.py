def update_image(self, values: torch.Tensor):
    if SafetensorRecorder.IMAGE_KEY in self.__data__:
        self.__data__[SafetensorRecorder.IMAGE_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.IMAGE_KEY] = [values]
