def update_seq(self, values: torch.Tensor):
    if SafetensorRecorder.SEQ_KEY in self.__data__:
        self.__data__[SafetensorRecorder.SEQ_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.SEQ_KEY] = [values]
