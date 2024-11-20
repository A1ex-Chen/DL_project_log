def batch_frames(self, batch_size: int=32, calibrate: bool=False, crop:
    bool=False) ->Generator[NDArray, None, None]:
    """Iterate over frames of video.

        Yields:
            NDArray: frame of video.
        """
    frames = []
    for frame in self:
        frames.append(frame)
        if len(frames) == batch_size:
            yield np.stack(frames)
            frames = []
    if len(frames) > 0:
        yield np.stack(frames)
