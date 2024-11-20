def strip_modelkit_traceback_frames(exc: BaseException):
    """
    Walk the traceback and remove frames that originate from within modelkit
    Return an exception with the filtered traceback
    """
    tb = None
    for tb_frame, _ in reversed(list(traceback.walk_tb(exc.__traceback__))):
        if not is_modelkit_internal_frame(tb_frame):
            tb = types.TracebackType(tb, tb_frame, tb_frame.f_lasti,
                tb_frame.f_lineno)
    return exc.with_traceback(tb)
