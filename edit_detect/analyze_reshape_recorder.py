def reshape_recorder(recorder: SafetensorRecorder, ts: int, size: int):
    recorder.__data__[SafetensorRecorder.IMAGE_KEY] = recorder.__data__[
        SafetensorRecorder.IMAGE_KEY].reshape(-1, 3, size, size)
    recorder.__data__[SafetensorRecorder.NOISE_KEY] = recorder.__data__[
        SafetensorRecorder.NOISE_KEY].reshape(-1, 3, size, size)
    recorder.__data__[SafetensorRecorder.NOISY_IMAGE_KEY] = recorder.__data__[
        SafetensorRecorder.NOISY_IMAGE_KEY].reshape(-1, 3, size, size)
    recorder.__data__[SafetensorRecorder.RECONST_KEY] = recorder.__data__[
        SafetensorRecorder.RECONST_KEY].reshape(-1, 3, size, size)
    recorder.__data__[SafetensorRecorder.LABEL_KEY] = recorder.__data__[
        SafetensorRecorder.LABEL_KEY].reshape(-1)
    if SafetensorRecorder.SEQ_KEY in recorder.__data__:
        recorder.__data__[SafetensorRecorder.SEQ_KEY] = recorder.__data__[
            SafetensorRecorder.SEQ_KEY].reshape(-1, ts, 3, size, size)
    if SafetensorRecorder.RESIDUAL_KEY in recorder.__data__:
        recorder.__data__[SafetensorRecorder.RESIDUAL_KEY] = recorder.__data__[
            SafetensorRecorder.RESIDUAL_KEY].reshape(-1, ts)
    if SafetensorRecorder.TRAJ_RESIDUAL_KEY in recorder.__data__:
        recorder.__data__[SafetensorRecorder.TRAJ_RESIDUAL_KEY
            ] = recorder.__data__[SafetensorRecorder.TRAJ_RESIDUAL_KEY
            ].reshape(-1, ts - 1)
    recorder.__data__[SafetensorRecorder.TS_KEY] = recorder.__data__[
        SafetensorRecorder.TS_KEY].reshape(-1)
    return recorder
