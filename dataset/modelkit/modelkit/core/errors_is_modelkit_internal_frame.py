def is_modelkit_internal_frame(frame: types.FrameType):
    """
    Guess whether the frame originates from a submodule of `modelkit`
    """
    try:
        mod = inspect.getmodule(frame)
        if mod:
            frame_package = __package__.split('.')[0]
            return frame_package == 'modelkit'
    except BaseException:
        pass
    return False
