def update_label(self, values: torch.Tensor):
    if SafetensorRecorder.LABEL_KEY in self.__data__:
        self.__data__[SafetensorRecorder.LABEL_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.LABEL_KEY] = [values]
