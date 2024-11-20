def update_traj_residual(self, values: torch.Tensor):
    if SafetensorRecorder.TRAJ_RESIDUAL_KEY in self.__data__:
        self.__data__[DirectRecorder.TRAJ_RESIDUAL_KEY].append(values)
    else:
        self.__data__[SafetensorRecorder.TRAJ_RESIDUAL_KEY] = [values]
