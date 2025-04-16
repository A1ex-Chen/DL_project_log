@contextlib.contextmanager
def exceptions_as_analysis_errors(project_root):
    try:
        yield
    except _SuspendExecution:
        pass
    except AnalysisError:
        raise
    except Exception as ex:
        logger.debug(
            "An error occured during analysis (could be a problem with the user's code):"
            , exc_info=ex)
        if isinstance(ex, SyntaxError):
            error = AnalysisError(
                'DeepView encountered a syntax error while profiling your model.'
                )
        else:
            error = AnalysisError(str(ex), type(ex))
        if hasattr(ex, 'filename') and ex.filename.startswith(project_root):
            _add_context_to_error(error, project_root, ex.filename, getattr
                (ex, 'lineno', None))
        else:
            stack = traceback.extract_tb(ex.__traceback__)
            for frame in reversed(stack):
                if frame.filename.startswith(project_root):
                    _add_context_to_error(error, project_root, frame.
                        filename, frame.lineno)
                    break
        if error.file_context is None and str(error).startswith(
            'TypeError: forward() takes'):
            error = AnalysisError(
                '{}. This error could be due to a mismatch between the number of inputs that your model expects and the number of inputs that your input provider returns.'
                .format(str(error)))
        raise error
